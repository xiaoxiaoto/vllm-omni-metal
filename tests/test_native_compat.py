# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

from vllm_omni_metal.native_compat import adapt_mps_capture_queries


class ValidatedModel(nn.Module):
    def validate(self, codes, *, limit=4):
        if not torch.cuda.is_current_stream_capturing():
            if codes.max().item() >= limit:
                raise ValueError("invalid code")
        return codes + 1

    @staticmethod
    def capture_status():
        return torch.cuda.is_current_stream_capturing()


def test_capture_compatibility_is_instance_local_and_keeps_validation():
    original = torch.cuda.is_current_stream_capturing
    cls_method = ValidatedModel.validate
    globals_before = cls_method.__globals__["torch"]
    model = ValidatedModel()
    adapt_mps_capture_queries(model)
    assert model.capture_status() is False
    torch.testing.assert_close(model.validate(torch.tensor([1]), limit=2), torch.tensor([2]))
    with pytest.raises(ValueError, match="invalid code"):
        model.validate(torch.tensor([4]))
    assert ValidatedModel.validate is cls_method
    assert cls_method.__globals__["torch"] is globals_before
    assert torch.cuda.is_current_stream_capturing is original
    assert ValidatedModel().validate.__func__ is cls_method
