from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from adetailer.common import PredictOutput, resolve_sam3_model_path


class SAM3DependencyError(RuntimeError):
    """Raised when SAM3 backend is selected but unavailable."""


def sam3_predict(
    image: Image.Image,
    *,
    text_prompt: str,
    model_name: str,
    model_dirs: list[str | Path] | None = None,
    confidence: float = 0.3,
) -> PredictOutput[float]:
    """Entry point for future SAM3 text-conditioned detection.

    This function is intentionally lightweight:
    - no SAM3 dependency import at module import time,
    - no automatic model download,
    - clear runtime error only when SAM3 is explicitly selected.
    """

    _ = confidence
    if not text_prompt.strip():
        raise SAM3DependencyError(
            "ADetailer SAM3 backend requires a non-empty text prompt. "
            "Set 'ADetailer detector classes' to your SAM3 text query."
        )

    model_path = resolve_sam3_model_path(model_name, *(model_dirs or []))
    if model_path is None:
        raise SAM3DependencyError(
            "ADetailer SAM3 backend is selected, but SAM3 model was not found. "
            f"model={model_name!r}. Install SAM3 manually and place the weight file in "
            "an ADetailer model directory or provide an absolute path."
        )

    # NOTE: keep optional import local so Forge startup is unaffected.
    try:
        import sam3  # type: ignore # noqa: F401
    except Exception as e:  # pragma: no cover - depends on local environment
        raise SAM3DependencyError(
            "ADetailer SAM3 backend is selected, but SAM3 Python package is not available. "
            "Install SAM3 dependencies manually to use this detector."
        ) from e

    raise NotImplementedError(
        "SAM3 backend scaffold is ready, but inference is not implemented yet. "
        "TODO: run SAM3 text-conditioned inference and convert results via "
        "sam3_results_to_predict_output()."
    )


def sam3_results_to_predict_output(
    *,
    image: Image.Image,
    boxes: list[list[float]] | None,
    masks: list[Image.Image | None] | None,
    scores: list[float] | None,
    labels: list[str] | None,
) -> PredictOutput[float]:
    """Convert future SAM3 raw outputs to ADetailer PredictOutput."""

    output = PredictOutput[float](
        bboxes=boxes or [],
        masks=masks or [],
        confidences=scores or [],
        labels=labels or [],
        preview=image,
    )
    return output
