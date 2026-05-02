from __future__ import annotations

from pathlib import Path
import types

import pytest
from PIL import Image

from adetailer.args import ADetailerArgs
from adetailer.common import resolve_sam3_model_path
from adetailer.sam3 import (
    SAM3DependencyError,
    _run_sam3_inference,
    sam3_predict,
    sam3_results_to_predict_output,
)


def test_args_is_sam3() -> None:
    assert ADetailerArgs(ad_model="sam3_text").is_sam3() is True
    assert ADetailerArgs(ad_model="face_yolov8n.pt").is_sam3() is False


def test_resolve_sam3_model_path_none_when_missing(tmp_path: Path) -> None:
    assert resolve_sam3_model_path("missing.pt", tmp_path) is None


def test_sam3_predict_missing_prompt() -> None:
    image = Image.new("RGB", (32, 32), "white")
    with pytest.raises(SAM3DependencyError, match="non-empty text prompt"):
        sam3_predict(image=image, text_prompt="", model_name="sam3_missing.pt")


def test_sam3_results_bbox_from_mask_and_selection() -> None:
    image = Image.new("RGB", (16, 16), "black")
    m1 = Image.new("L", (16, 16), 0)
    m2 = Image.new("L", (16, 16), 0)
    for x in range(1, 5):
        for y in range(1, 5):
            m1.putpixel((x, y), 255)
    for x in range(10, 15):
        for y in range(2, 10):
            m2.putpixel((x, y), 255)

    out = sam3_results_to_predict_output(
        image=image,
        boxes=None,
        masks=[m1, m2],
        scores=[0.7, 0.8],
        labels=["a", "b"],
        target_selection="largest",
    )
    assert len(out.bboxes) == 1
    assert out.labels == ["b"]


def test_sam3_results_drop_empty_and_min_area() -> None:
    image = Image.new("RGB", (8, 8), "black")
    empty = Image.new("L", (8, 8), 0)
    small = Image.new("L", (8, 8), 0)
    small.putpixel((0, 0), 255)
    out = sam3_results_to_predict_output(
        image=image,
        boxes=None,
        masks=[empty, small],
        scores=None,
        labels=None,
        min_mask_area=2,
    )
    assert out.bboxes == []


def test_run_sam3_inference_official_api_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_builder = types.ModuleType("sam3.model_builder")
    fake_processor_mod = types.ModuleType("sam3.model.sam3_image_processor")

    def build_sam3_image_model(checkpoint_path: str):
        return {"checkpoint": checkpoint_path}

    class FakeProcessor:
        def __init__(self, model):
            self.model = model

        def set_image(self, image):
            return {"image": image}

        def set_text_prompt(self, state, prompt):
            m = Image.new("L", (8, 8), 255)
            return {"masks": [m], "boxes": [[0, 0, 7, 7]], "scores": [0.9]}

    fake_builder.build_sam3_image_model = build_sam3_image_model
    fake_processor_mod.Sam3Processor = FakeProcessor
    monkeypatch.setitem(__import__("sys").modules, "sam3.model_builder", fake_builder)
    monkeypatch.setitem(
        __import__("sys").modules, "sam3.model.sam3_image_processor", fake_processor_mod
    )

    out = _run_sam3_inference(
        sam3=object(),
        image=__import__("numpy").zeros((8, 8, 3), dtype="uint8"),
        text_prompt="girl face",
        model_path="models/sam3/sam3.pt",
        confidence=0.3,
    )
    assert len(out[0]) == 1
