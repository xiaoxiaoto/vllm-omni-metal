# SPDX-License-Identifier: Apache-2.0
"""ABC wire/termination regressions; no engine, HTTP, weights, or audio generation."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

DEPLOY = Path(__file__).resolve().parents[2] / "docker-compose" / "yue2-3b"
pytestmark = pytest.mark.skipif(
    not (DEPLOY / "yue2_planning.py").is_file(),
    reason="independent YuE2 deployment checkout is not present",
)


@pytest.fixture
def planning():
    spec = importlib.util.spec_from_file_location("yue2_planning", DEPLOY / "yue2_planning.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native(planning, monkeypatch):
    # Wire fixtures only: generated ids below are not model-generation evidence.
    model = SimpleNamespace(EOD=151643, ABC_END=151848, DEFAULT_SEED=831001)
    helper = Mock(side_effect=lambda ids: ids[:-1] if ids[-1] == model.ABC_END else list(ids))
    result = SimpleNamespace(
        model=model,
        prompt=SimpleNamespace(abc_ids_from_generated=helper),
        tokenizer=SimpleNamespace(decode=Mock(return_value="X:1\nK:C\nCDEF|")),
    )
    monkeypatch.setattr(planning, "_load_native", lambda _: result)
    return result


def response(ids=None, *, reason="stop", stop=151848, **kwargs):
    return {
        "choices": [
            {
                "token_ids": [10, 20, 151848] if ids is None else ids,
                "text": "not the checkpoint tokenizer's authoritative decode",
                "finish_reason": reason,
                "stop_reason": stop,
                **kwargs,
            }
        ]
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"cot": "off"},
        {"style": ""},
        {"lyrics": "  "},
        {"seed": True},
        {"seed": "12300"},
        {"seed": 2**63},
        {"seed": -(2**63) - 1},
    ],
)
def test_invalid_input_is_not_silently_changed(planning, native, changes):
    with pytest.raises(ValueError):
        planning.build_abc_request("/local/model", {"style": "pop", "lyrics": "song", **changes})


@pytest.mark.parametrize(
    "ids,stop",
    [
        ([10, 20, 151848], 151848),
        ([10, 20], 151848),  # Transport omitted END; explicit stop reason still proves it.
        ([10, 20, 151848], None),  # Raw terminal token is sufficient with finish=stop.
    ],
)
def test_matching_stop_evidence_accepts_only_the_ordinary_text_span(planning, native, ids, stop):
    original = response(ids, stop=stop)
    result = planning.extract_abc(original, "/local/model")
    assert result["naturally_ended"] is True
    assert result["abc_token_ids"] == [10, 20]


@pytest.mark.parametrize(
    "reply",
    [
        response(reason="length"),
        response(reason="abort"),
        response(reason=None),
        response([10, 20], stop=None),
        response(stop=151852),
        response(stop="151848"),
        response(stop=True),
        response([]),
        response([151848]),
        response([-1, 151848]),
        response([151643, 151848]),
        response([151847, 10, 151848]),
        response([10, 151848, 20]),
        response([10, True, 151848]),
        response([10.0, 151848]),
        response([10, 151848], truncated=True),
        {"truncated": True, **response()},
        {"error": {"message": "engine failed"}, **response()},
        {"choices": []},
        {"choices": [{}, {}]},
        {"choices": [None]},
        {"choices": [{"finish_reason": "stop", "stop_reason": 151848, "text": "X:1\nK:C\nCDEF|"}]},
    ],
)
def test_invalid_plan_is_rejected_without_forging_end(planning, native, reply):
    before = json.dumps(reply)
    with pytest.raises(ValueError):
        planning.extract_abc(reply, "/local/model")
    assert json.dumps(reply) == before
    native.tokenizer.decode.assert_not_called()


def test_whitespace_score_is_not_a_plan(planning, native):
    native.tokenizer.decode.return_value = " \n\t"
    with pytest.raises(ValueError, match="empty text"):
        planning.extract_abc(response(), "/local/model")


@pytest.fixture
def installed_native(planning):
    model_path = Path(
        os.environ.get(
            "YUE2_MODEL_PATH", Path.home() / ".cache/modelscope/hub/models/m-a-p/YuE2-3B"
        )
    ).expanduser()
    if not (model_path / "qwen.tiktoken").is_file():
        pytest.skip("local checkpoint-native qwen.tiktoken is unavailable")
    try:
        native = planning._load_native(str(model_path))
    except ModuleNotFoundError as exc:
        pytest.skip(f"selected YuE2 overlay dependencies unavailable: {exc}")
    return model_path, native


def test_real_checkpoint_tokenizer_decodes_abc_not_http_text(planning, installed_native):
    model_path, native = installed_native
    score = 'X:1\nT:今晚不眠\nM:4/4\nL:1/8\nK:C\n"C"CDEF GABc|\n'
    ids = native.tokenizer.encode(score)
    reply = response(ids + [native.model.ABC_END], stop=native.model.ABC_END)
    reply["choices"][0]["text"] = ""  # skip_tokenizer_init deployment contract.
    result = planning.extract_abc(reply, model_path)
    assert result["abc"] == score
    assert result["abc_token_ids"] == ids
