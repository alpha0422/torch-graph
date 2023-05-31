#!/usr/bin/env python

import torch
import time
from functorch.compile import aot_module, make_boxed_func
from torch._functorch.partitioners import draw_graph

__all__ = ["capture"]

def my_compiler(fx_module: torch.fx.GraphModule, _):
    draw_graph(fx_module, f"aot.{time.time()}.svg")
    return make_boxed_func(fx_module.forward)

def capture(model, *inputs):
    aot_model = aot_module(model, fw_compiler=my_compiler)
    y = aot_model(*inputs)
    y.sum().backward()

if __name__ == '__main__':
    import torch.nn as nn
    model = nn.Sequential(
        nn.Conv2d(16, 32, 3),
        nn.BatchNorm2d(32),
        nn.SiLU(),
    ).cuda()
    x = torch.randn((2, 16, 8, 8), requires_grad=True, device="cuda")

    capture(model, x)

