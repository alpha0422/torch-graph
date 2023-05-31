#!/usr/bin/env python

from torchgraph import dispatch_capture, aot_capture, compile_capture

import torch
import torch.nn as nn
model = nn.Sequential(
    nn.Conv2d(16, 32, 3),
    nn.BatchNorm2d(32),
    nn.SiLU(),
).cuda()
x = torch.randn((2, 16, 8, 8), requires_grad=True, device="cuda")


dispatch_capture(model, x)
aot_capture(model, x)
compile_capture(model, x)
