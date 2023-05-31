#!/usr/bin/env python

import torch
import time
from torch._functorch.partitioners import draw_graph

__all__ = ["capture"]

def my_compiler(fx_module: torch.fx.GraphModule, _):
    draw_graph(fx_module, f"compile.{time.time()}.svg")
    return fx_module.forward

def capture(model, *inputs):
    compiled_model = torch.compile(model, backend=my_compiler)
    y = compiled_model(*inputs)
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

