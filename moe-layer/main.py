from moe import MOEFeedForward, mixtral_8x7b_v0_1_config
import torch
from dist import init_dist, shard
import copy
import nvtx # type: ignore

if __name__ == "__main__":
    config = mixtral_8x7b_v0_1_config
    torch.manual_seed(2)  # type: ignore
    init_dist()
    model = MOEFeedForward.random(config)
    model.to("cuda")
    model_tp = copy.deepcopy(model)
    model_tp.apply_tp()
    model_ep = copy.deepcopy(model)
    model_ep.apply_ep()
    
    # print(model)
    # print(model_tp)
    # print(model_ep)
    # # print("model", model(x))
    # # print("model_tp", model_tp(x))
    # print(x)
    # print(shard(x, 1))
    # print("model", model(x))
    
    for i in range(20):
        x = torch.randn(
            1, 512, config.dim, dtype=torch.bfloat16, device="cuda"
        )
        sharded = shard(x, 1)
        print("start")
        with nvtx.annotate("step"):
            model_ep(sharded)
        print("ok")
    # print("model_ep", model_ep(shard(x, 1)))
    # print("model_ep", model_ep(shard(x, 1)))
    # print("model_ep", model_ep(shard(x, 1)))
    # print("model_ep", model_ep(shard(x, 1)))
