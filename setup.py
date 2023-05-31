#!/usr/bin/env python

from setuptools import setup

setup(
    name="torchgraph",
    version="0.1",
    description="PyTorch graph capturing.",
    author="Fei Kong",
    author_email="alpha0422@gmail.com",
    url="https://github.com/alpha0422/torch-graph",
    packages=["torchgraph"],
    install_requires=[
        "graphviz",
        "pydot",
    ],
)
