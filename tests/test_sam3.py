from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from adetailer.args import ADetailerArgs
from adetailer.common import resolve_sam3_model_path
from adetailer.sam3 import SAM3DependencyError, sam3_predict


def test_args_is_sam3() -> None:
    assert ADetailerArgs(ad_model="sam3_text").is_sam3() is True
    assert ADetailerArgs(ad_model="face_yolov8n.pt").is_sam3() is False


def test_resolve_sam3_model_path_none_when_missing(tmp_path: Path) -> None:
    assert resolve_sam3_model_path("missing.pt", tmp_path) is None


def test_sam3_predict_missing_prompt() -> None:
    image = Image.new("RGB", (32, 32), "white")
    with pytest.raises(SAM3DependencyError, match="non-empty text prompt"):
        sam3_predict(image=image, text_prompt="", model_name="sam3_missing.pt")
