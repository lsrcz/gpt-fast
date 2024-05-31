from dataclasses import dataclass
import nvtx  # type: ignore
from typing import Self, Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.distributed as dist
from dist import shard, get_rank, get_world_size


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    # block_size: int = 2048
    # vocab_size: int = 32000
    # n_layer: int = 32
    # n_head: int = 32
    dim: int = 4096
    intermediate_size: int = -1
    # n_local_heads: int = -1
    # head_dim: int = 64
    # rope_base: float = 10000
    # norm_eps: float = 1e-5
    num_experts: int = 8
    num_activated_experts: int = 2

    def __post_init__(self):
        # if self.n_local_heads == -1:
        #    self.n_local_heads = self.n_head
        if self.intermediate_size == -1:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        # self.head_dim = self.dim // self.n_head


mixtral_8x7b_v0_1_config: ModelArgs = ModelArgs(
    # block_size=32768,
    # n_layer=32,
    # n_head=32,
    # n_local_heads=8,
    dim=4096,
    intermediate_size=14336,
    # rope_base=1000000.0,
    num_experts=8,
    num_activated_experts=2,
)

small_config: ModelArgs = ModelArgs(
    dim=4,
    intermediate_size=16,
    num_experts=8,
    num_activated_experts=2,
)


class MOEFeedForwardGating(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()  # type: ignore
        self.gate = nn.Linear(
            config.dim, config.num_experts, bias=False, dtype=torch.bfloat16
        )
        self.num_activated_experts = config.num_activated_experts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores: torch.Tensor = self.gate(x)
        expert_weights = F.softmax(scores, dim=-1)
        return torch.topk(expert_weights, self.num_activated_experts, dim=-1)


class MOEFeedForwardAfterGating(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()  # type: ignore
        self.w1 = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.intermediate_size,
                config.dim,
                requires_grad=False,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.w2 = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.dim,
                config.intermediate_size,
                requires_grad=False,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.w3 = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.intermediate_size,
                config.dim,
                requires_grad=False,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.num_activated_experts = config.num_activated_experts
        self.num_experts = config.num_experts
        self.forward_func = self._forward_single

    def _forward_single(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)  # [T, A]
        w1_weights = self.w1[expert_indices]  # [T, A, D, D]
        w3_weights = self.w3[expert_indices]  # [T, A, D, D]
        w2_weights = self.w2[expert_indices]  # [T, A, D, D]
        x1 = F.silu(torch.einsum("ti,taoi -> tao", x, w1_weights))
        x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
        expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
        return torch.einsum("tai,ta -> ti", expert_outs, expert_weights)

    def _forward_tp(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)
        w1_weights = self.w1[expert_indices]
        w3_weights = self.w3[expert_indices]
        w2_weights = self.w2[expert_indices]  # [T, A, D, D]
        x1 = F.silu(torch.einsum("ti,taoi -> tao", x, w1_weights))
        x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
        expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
        dist.all_reduce(expert_outs)  # type: ignore
        return torch.einsum("tai,ta -> ti", expert_outs, expert_weights)

    def apply_tp(self) -> None:
        self.w1 = nn.Parameter(shard(self.w1, 1), requires_grad=False)
        self.w3 = nn.Parameter(shard(self.w3, 1), requires_grad=False)
        self.w2 = nn.Parameter(shard(self.w2, 2), requires_grad=False)
        self.forward_func = self._forward_tp

    def _forward_ep(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with expert parallelism. The input are for part of the
        inputs, we need to do an all to all dispatch to first dispatch the
        inputs to the correct experts, then do an all-to-all combine to collect
        the outputs from all experts."""
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)
        concated_x = torch.cat([x] * self.num_activated_experts)
        concated_expert_indices = torch.reshape(
            torch.transpose(expert_indices, 0, 1), (-1,)
        )

        with nvtx.annotate("shape"):
        # Compute the shape of the buffers
            step = self.num_experts // get_world_size()
            start_m_all = (
                torch.arange(0, self.num_experts, step)
                .reshape(-1, 1)
                .expand(-1, concated_expert_indices.size(0))
            ).to("cuda")
            end_m_all = (
                torch.arange(step, self.num_experts + 1, step)
                .reshape(-1, 1)
                .expand(-1, concated_expert_indices.size(0))
            ).to("cuda")
            assert start_m_all.size(0) == get_world_size()
            assert end_m_all.size(0) == get_world_size()
            expert_mask = torch.logical_and(
                concated_expert_indices >= start_m_all,
                concated_expert_indices < end_m_all,
            )
            num_for_rank = expert_mask.count_nonzero(dim=1)

            # All to all to dispatch the shape of the buffers
            num_from_rank = [
                torch.zeros(1, dtype=torch.int64).to("cuda")
                for _ in range(get_world_size())
            ]
            dist.all_to_all(num_from_rank, list(num_for_rank))  # type: ignore

        with nvtx.annotate("dispatch"):
            # All to all to dispatch the inputs
            x_all_to_all_buffers = [
                torch.zeros(
                    int(num_from_rank[i]), x.size(-1), dtype=torch.bfloat16
                ).to("cuda")
                for i in range(get_world_size())
            ]
            x_all_to_all_to_send = [
                concated_x[expert_mask[i]] for i in range(get_world_size())
            ]
            dist.all_to_all(x_all_to_all_buffers, x_all_to_all_to_send)  # type: ignore

            expert_indices_all_to_all_buffers = [
                torch.zeros(int(num_from_rank[i]), dtype=torch.int64).to("cuda")
                for i in range(get_world_size())
            ]
            expert_indices_to_send = [
                concated_expert_indices[expert_mask[i]]
                for i in range(get_world_size())
            ]
            dist.all_to_all(  # type: ignore
                expert_indices_all_to_all_buffers, expert_indices_to_send
            )
            # print(f"in rank{get_rank()}, x: {x}")
            # print(f"in rank{get_rank()}, expert_weights: {expert_weights}")
            # print(f"in rank{get_rank()}, expert_indices: {expert_indices}")
            # print(f"in rank{get_rank()}, expert mask: {expert_mask}")
            # print(f"in rank{get_rank()}, x_all_to_all_to_send: {x_all_to_all_to_send}")
            # print(
                # f"in rank{get_rank()}, x_all_to_all_buffers: {x_all_to_all_buffers}"
            # )
            # print(
                # f"in rank{get_rank()}, expert_indices_all_to_all_buffers: {expert_indices_all_to_all_buffers}"
            # )

            all_x_for_current_expert = torch.cat(x_all_to_all_buffers, dim=0)
            # print(
            #     f"in rank{get_rank()}, all_x_for_current_expert: {all_x_for_current_expert}"
            # )
            all_expert_indices_for_current_expert = (
                torch.cat(expert_indices_all_to_all_buffers, dim=0)
                - step * get_rank()
            )
            # print(
            #     f"in rank{get_rank()}, all_expert_indices_for_current_expert: {all_expert_indices_for_current_expert}"
            # )

        with nvtx.annotate("compute"):
            w1_weights = self.w1[all_expert_indices_for_current_expert]
            w3_weights = self.w3[all_expert_indices_for_current_expert]
            w2_weights = self.w2[all_expert_indices_for_current_expert]

            x1 = F.silu(
                torch.einsum("ti,toi -> to", all_x_for_current_expert, w1_weights)
            )
            x3 = torch.einsum("ti, toi -> to", all_x_for_current_expert, w3_weights)
            expert_outs = torch.einsum("to, tio -> ti", (x1 * x3), w2_weights)
            # print(f"in rank{get_rank()}, expert_outs: {expert_outs}")

        with nvtx.annotate("backdispatch"):
            # All to all to dispatch the outputs
            expert_outs_buffer = torch.zeros_like(concated_x)
            expert_outs_output_split = [
                x_all_to_all_to_send[i].size(0) for i in range(get_world_size())
            ]
            expert_outs_input_split = [
                x_all_to_all_buffers[i].size(0) for i in range(get_world_size())
            ]
            # print(
            #     f"in rank{get_rank()}, expert_outs_buffer.shape: {expert_outs_buffer.shape}"
            # )
            # print(f"in rank{get_rank()}, expert_outs.shape: {expert_outs.shape}")
            # print(
            #     f"in rank{get_rank()}, expert_outs_output_split: {expert_outs_output_split}"
            # )
            # print(
            #     f"in rank{get_rank()}, expert_outs_input_split: {expert_outs_input_split}"
            # )
            dist.all_to_all_single(  # type: ignore
                expert_outs_buffer,
                expert_outs,
                expert_outs_output_split,
                expert_outs_input_split,
            )
            # print(f"in rank{get_rank()}, expert_outs_buffer: {expert_outs_buffer}")
            # print(f"in rank{get_rank()}, expert_mask: {expert_mask}")

            def exclusive_cumsum_1(t: torch.Tensor) -> torch.Tensor:
                out = torch.cumsum(t, dim=1).roll(1, 1)
                out[:, 0] = 0
                return out

            def exclusive_cumsum_0(t: torch.Tensor) -> torch.Tensor:
                out = torch.cumsum(t, dim=0).roll(1, 0)
                out[0] = 0
                return out

            reverse_perm_map = (
                (
                    exclusive_cumsum_1(expert_mask.int())
                    + exclusive_cumsum_0(num_for_rank).reshape(-1, 1)
                )
                .masked_fill(expert_mask.logical_not(), 0)
                .sum(dim=0)
                .reshape(self.num_activated_experts, -1).transpose(0, 1)
            )
            # print(f"in rank{get_rank()}, reverse_perm_map: {reverse_perm_map}")

            current_expert_outputs = expert_outs_buffer[reverse_perm_map]
            # print(f"in rank{get_rank()}, current_expert_outputs: {current_expert_outputs}")

        return torch.einsum("tai,ta -> ti", current_expert_outputs, expert_weights)

    def apply_ep(self) -> None:
        self.w1 = nn.Parameter(shard(self.w1, 0), requires_grad=False)
        self.w3 = nn.Parameter(shard(self.w3, 0), requires_grad=False)
        self.w2 = nn.Parameter(shard(self.w2, 0), requires_grad=False)
        self.forward_func = self._forward_ep

    def forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_func(x, expert_weights, expert_indices)


class MOEFeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()  # type: ignore
        self.gate = MOEFeedForwardGating(config)
        self.after_gate = MOEFeedForwardAfterGating(config)
        self.dim = config.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.dim)
        expert_weights, expert_indices = self.gate(x)
        return self.after_gate(x, expert_weights, expert_indices)

    def apply_tp(self) -> None:
        self.after_gate.apply_tp()

    def apply_ep(self) -> None:
        self.after_gate.apply_ep()

    @classmethod
    def random(cls, config: ModelArgs) -> Self:
        zero = cls(config)
        zero.gate.gate.requires_grad_(False)
        zero.gate.gate.weight.random_(-10, 10)
        zero.after_gate.w1.random_(-10, 10)
        zero.after_gate.w2.random_(-10, 10)
        zero.after_gate.w3.random_(-10, 10)
        return zero
