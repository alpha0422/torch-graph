# torch-graph

Simple PyTorch graph capturing.

## Instructions

Please install `graphviz` first:

```console
$ apt-get install graphviz
```

Clone and install this package:

```console
$ pip install .
```

Examples:

```python
from torchgraph import dispatch_capture, aot_capture, compile_capture

import torch
import torch.nn as nn
model = nn.Sequential(
    nn.Conv2d(16, 32, 3),
    nn.BatchNorm2d(32),
    nn.SiLU(),
).cuda()
x = torch.randn((2, 16, 8, 8), requires_grad=True, device="cuda")

# Capture joint forward and backward graph through dispatch
dispatch_capture(model, x)

# Capture separate forward and backward graphs through PyTorch AOTAutograd
aot_capture(model, x)

# Capture forward graph through PyTorch compile
compile_capture(model, x)
```

You'll find the captured graphs in `.svg` format under current folder.

