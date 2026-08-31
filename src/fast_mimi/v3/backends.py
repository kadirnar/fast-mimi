from __future__ import annotations
import torch
from .model import FastMimi
from .graphs import GraphedMimi


def build(name: str, state, dtype=torch.bfloat16, **kw):
    if name == "torch":
        return FastMimi(state, dtype=dtype, **kw)
    if name == "graph":
        return GraphedMimi(FastMimi(state, dtype=dtype, **kw))
    if name.startswith("triton"):
        from .triton_backend import TritonMimi
        m = TritonMimi(state, dtype=dtype, variant=name, **kw)
        return m
    if name.startswith("graph+"):
        return GraphedMimi(build(name[len("graph+"):], state, dtype, **kw))
    raise ValueError(name)
