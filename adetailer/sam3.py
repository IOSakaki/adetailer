from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
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
    target_selection: str = "all",
    min_mask_area: int = 0,
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
        import sam3  # type: ignore
    except Exception as e:  # pragma: no cover - depends on local environment
        raise SAM3DependencyError(
            "ADetailer SAM3 backend is selected, but SAM3 Python package is not available. "
            "Install SAM3 dependencies manually to use this detector."
        ) from e

    np_image = np.array(image.convert("RGB"))
    masks, boxes, scores, labels = _run_sam3_inference(
        sam3=sam3,
        image=np_image,
        text_prompt=text_prompt,
        model_path=model_path,
        confidence=confidence,
    )
    return sam3_results_to_predict_output(
        image=image,
        boxes=boxes,
        masks=masks,
        scores=scores,
        labels=labels,
        target_selection=target_selection,
        min_mask_area=min_mask_area,
    )


def _run_sam3_inference(
    *, sam3: Any, image: np.ndarray, text_prompt: str, model_path: str, confidence: float
):
    # Official README path:
    #   from sam3.model_builder import build_sam3_image_model
    #   from sam3.model.sam3_image_processor import Sam3Processor
    #   output = processor.set_text_prompt(...)
    try:
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
        from sam3.model_builder import build_sam3_image_model  # type: ignore

        pil_image = Image.fromarray(image)
        model = build_sam3_image_model(checkpoint_path=model_path)
        processor = Sam3Processor(model)
        state = processor.set_image(pil_image)
        output = processor.set_text_prompt(state=state, prompt=text_prompt)
        return (
            output.get("masks", []),
            output.get("boxes", []),
            output.get("scores", []),
            [text_prompt] * len(output.get("masks", [])),
        )
    except Exception as official_error:
        # compatibility fallback for API moves/renames in SAM3 variants
        fallback_output, fallback_error = _run_sam3_fallback(
            sam3, image, text_prompt, model_path, confidence
        )
        if fallback_output is not None:
            return fallback_output
        raise SAM3DependencyError(
            "SAM3 package was found but image text-prompt inference API could not be called. "
            "Tried official Sam3Processor path first and then compatibility fallbacks. "
            f"Official error: {official_error!r}; fallback error: {fallback_error!r}"
        ) from official_error


def _run_sam3_fallback(
    sam3: Any, image: np.ndarray, text_prompt: str, model_path: str, confidence: float
) -> tuple[tuple[Any, Any, Any, Any] | None, Exception | None]:
    try:
        if hasattr(sam3, "predict_text_masks"):
            return sam3.predict_text_masks(
                image=image, prompt=text_prompt, checkpoint=model_path, confidence=confidence
            ), None
        if hasattr(sam3, "text_mask_predict"):
            return sam3.text_mask_predict(
                image=image, text=text_prompt, checkpoint=model_path, conf=confidence
            ), None
    except Exception as e:  # pragma: no cover
        return None, e
    return None, RuntimeError("no compatible fallback API found")


def sam3_results_to_predict_output(
    *,
    image: Image.Image,
    boxes: list[list[float]] | None,
    masks: list[Image.Image | None] | None,
    scores: list[float] | None,
    labels: list[str] | None,
    target_selection: str = "all",
    min_mask_area: int = 0,
) -> PredictOutput[float]:
    """Convert future SAM3 raw outputs to ADetailer PredictOutput."""

    out_boxes: list[list[float]] = []
    out_masks: list[Image.Image] = []
    out_scores: list[float] = []
    out_labels: list[str] = []
    in_masks = masks or []
    in_boxes = boxes or []
    in_scores = scores or []
    in_labels = labels or []

    for i, mask in enumerate(in_masks):
        if mask is None:
            continue
        m = mask.convert("L").resize(image.size)
        arr = np.array(m, dtype=np.uint8)
        bin_mask = arr > 0
        area = int(bin_mask.sum())
        if area <= 0 or area < min_mask_area:
            continue
        bbox = in_boxes[i] if i < len(in_boxes) else _bbox_from_mask(bin_mask)
        if bbox is None:
            continue
        out_masks.append(Image.fromarray((bin_mask * 255).astype(np.uint8), mode="L"))
        out_boxes.append([float(x) for x in bbox])
        out_scores.append(float(in_scores[i] if i < len(in_scores) else 1.0))
        out_labels.append(str(in_labels[i] if i < len(in_labels) else "sam3_text"))

    selected = _select_indices(out_boxes, out_masks, target_selection)
    output = PredictOutput[float](
        bboxes=[out_boxes[i] for i in selected],
        masks=[out_masks[i] for i in selected],
        confidences=[out_scores[i] for i in selected],
        labels=[out_labels[i] for i in selected],
        preview=image,
    )
    return output


def _bbox_from_mask(bin_mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(bin_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _select_indices(boxes: list[list[float]], masks: list[Image.Image], target_selection: str) -> list[int]:
    if not boxes:
        return []
    if target_selection == "all":
        return list(range(len(boxes)))
    if target_selection == "largest":
        areas = [int((np.array(m) > 0).sum()) for m in masks]
        return [int(np.argmax(areas))]
    centers = [((b[0] + b[2]) / 2.0) for b in boxes]
    if target_selection == "leftmost":
        return [int(np.argmin(centers))]
    if target_selection == "rightmost":
        return [int(np.argmax(centers))]
    return list(range(len(boxes)))
