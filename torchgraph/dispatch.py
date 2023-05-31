#!/usr/bin/env python

import time
import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakIdKeyDictionary
from graphviz import Digraph

__all__ = ["capture"]

class CaptureGraph(TorchDispatchMode):
    def __init__(self, fname="graph.dot"):
        self.fname = fname
        self._graph = Digraph(format="svg")
        self._tensors = WeakIdKeyDictionary()
        self._n_tensors = 0
        self._n_ops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **kwargs)
        op = f"{func}_{self._n_ops}"
        self._n_ops += 1
        self._graph.node(op, str(func), fillcolor="green")
        self._add_to_graph((args, kwargs), op, is_in=True)
        self._add_to_graph(out, op, is_in=False)
        return out

    def _add_to_graph(self, args, op, is_in=True):
        flat_args, _ = pytree.tree_flatten(args)
        for t in flat_args:
            if not torch.is_tensor(t):
                continue
            if t not in self._tensors:
                tensor = f"tensor_{self._n_tensors}"
                self._graph.node(tensor, fillcolor="skyblue")
                self._tensors[t] = tensor
                self._n_tensors += 1
            else:
                tensor = self._tensors[t]
            if is_in:
                self._graph.edge(tensor, op)
            else:
                self._graph.edge(op, tensor)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._graph.render(self.fname)

def capture(model, *inputs):
    primals = [p for p in model.parameters() if p.requires_grad]
    primals.extend([p for p in inputs if torch.is_tensor(p) and p.requires_grad])
    with CaptureGraph(f"dispatch.{time.time()}.dot"):
        loss = model(*inputs).sum()
        grads = torch.autograd.grad(loss, primals)

if __name__ == '__main__':
    import torch.nn as nn
    model = nn.Sequential(
        nn.Conv2d(16, 32, 3),
        nn.BatchNorm2d(32),
        nn.SiLU(),
    ).cuda()
    x = torch.randn((2, 16, 8, 8), requires_grad=True, device="cuda")

    capture(model, x)

