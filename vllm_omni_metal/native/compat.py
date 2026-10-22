# SPDX-License-Identifier: Apache-2.0
"""Instance-local compatibility for CUDA-only capture status queries on MPS."""

import inspect
from types import FunctionType, MethodType

import torch


class _MPSCudaQueries:
    @staticmethod
    def is_current_stream_capturing():
        return False

    def __getattr__(self, name):
        return getattr(torch.cuda, name)


class _MPSTorchView:
    cuda = _MPSCudaQueries()

    def __getattr__(self, name):
        return getattr(torch, name)


def adapt_mps_capture_queries(model):
    """Keep upstream validation active without touching module or torch globals."""
    for layer in model.modules():
        for name, method in inspect.getmembers(type(layer), inspect.isfunction):
            if "is_current_stream_capturing" not in method.__code__.co_names:
                continue
            if method.__globals__.get("torch") is not torch:
                continue
            namespace = dict(method.__globals__, torch=_MPSTorchView())
            clone = FunctionType(
                method.__code__, namespace, method.__name__, method.__defaults__, method.__closure__
            )
            clone.__kwdefaults__ = method.__kwdefaults__
            clone.__annotations__ = method.__annotations__
            clone.__dict__.update(method.__dict__)
            bound = getattr(layer, name)
            setattr(layer, name, MethodType(clone, layer) if inspect.ismethod(bound) else clone)
