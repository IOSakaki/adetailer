from __future__ import annotations

from collections import UserList
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, partial
from typing import Any, Literal, NamedTuple, Optional

try:
    from pydantic.v1 import (
        BaseModel,
        Extra,
        NonNegativeFloat,
        NonNegativeInt,
        PositiveInt,
        confloat,
        conint,
        validator,
    )
except ImportError:
    from pydantic import (
        BaseModel,
        Extra,
        NonNegativeFloat,
        NonNegativeInt,
        PositiveInt,
        confloat,
        conint,
        validator,
    )


@dataclass
class SkipImg2ImgOrig:
    steps: int
    sampler_name: str
    width: int
    height: int


class Arg(NamedTuple):
    attr: str
    name: str


class ArgsList(UserList):
    @cached_property
    def attrs(self) -> tuple[str, ...]:
        return tuple(attr for attr, _ in self)

    @cached_property
    def names(self) -> tuple[str, ...]:
        return tuple(name for _, name in self)


class ADetailerArgs(BaseModel, extra=Extra.forbid):
    ad_model: str = "None"
    ad_model_classes: str = ""
    ad_tab_enable: bool = True
    ad_prompt: str = ""
    ad_prompt_append: str = ""
    ad_negative_prompt: str = ""
    ad_negative_prompt_append: str = ""
    ad_confidence: confloat(ge=0.0, le=1.0) = 0.3
    ad_mask_filter_method: Literal["Area", "Confidence"] = "Area"
    ad_mask_k: NonNegativeInt = 0
    ad_mask_min_ratio: confloat(ge=0.0, le=1.0) = 0.0
    ad_mask_max_ratio: confloat(ge=0.0, le=1.0) = 1.0
    ad_dilate_erode: int = 4
    ad_mask_bbox_expansion: NonNegativeInt = 0
    ad_x_offset: int = 0
    ad_y_offset: int = 0
    ad_mask_merge_invert: Literal["None", "Merge", "Merge and Invert"] = "None"
    ad_mask_blur: NonNegativeInt = 4
    ad_denoising_strength: confloat(ge=0.0, le=1.0) = 0.4
    ad_inpaint_masked_content: Literal[
        "fill", "original", "latent noise", "latent nothing"
    ] = "original"
    ad_inpaint_only_masked: bool = True
    ad_inpaint_only_masked_padding: NonNegativeInt = 32
    ad_use_inpaint_width_height: bool = False
    ad_inpaint_width: PositiveInt = 512
    ad_inpaint_height: PositiveInt = 512
    ad_use_steps: bool = False
    ad_steps: PositiveInt = 28
    ad_use_cfg_scale: bool = False
    ad_cfg_scale: NonNegativeFloat = 7.0
    ad_use_checkpoint: bool = False
    ad_checkpoint: Optional[str] = None
    ad_use_vae: bool = False
    ad_vae: Optional[str] = None
    ad_use_sampler: bool = False
    ad_sampler: str = "DPM++ 2M Karras"
    ad_scheduler: str = "Use same scheduler"
    ad_use_noise_multiplier: bool = False
    ad_noise_multiplier: confloat(ge=0.5, le=1.5) = 1.0
    ad_use_clip_skip: bool = False
    ad_clip_skip: conint(ge=1, le=12) = 1
    ad_restore_face: bool = False
    ad_controlnet_model: str = "None"
    ad_controlnet_module: str = "None"
    ad_controlnet_weight: confloat(ge=0.0, le=1.0) = 1.0
    ad_controlnet_guidance_start: confloat(ge=0.0, le=1.0) = 0.0
    ad_controlnet_guidance_end: confloat(ge=0.0, le=1.0) = 1.0
    ad_controlnet_use_crop_input: bool = True
    is_api: bool = True

    @validator("is_api", pre=True)
    def is_api_validator(cls, v: Any):  # noqa: N805
        "tuple is json serializable but cannot be made with json deserialize."
        return type(v) is not tuple

    @staticmethod
    def ppop(
        p: dict[str, Any],
        key: str,
        pops: list[str] | None = None,
        cond: Any = None,
    ) -> None:
        if pops is None:
            pops = [key]
        if key not in p:
            return
        value = p[key]
        cond = (not bool(value)) if cond is None else value == cond

        if cond:
            for k in pops:
                p.pop(k, None)

    def extra_params(self, suffix: str = "") -> dict[str, Any]:
        if self.need_skip():
            return {}

        p = {name: getattr(self, attr) for attr, name in ALL_ARGS}
        ppop = partial(self.ppop, p)

        ppop("ADetailer Custom model classes")
        ppop("ADetailer Custom prompt")
        ppop("ADetailer Custom prompt append")
        ppop("ADetailer Custom negative prompt")
        ppop("ADetailer Custom negative prompt append")
        p.pop("ADetailer Custom tab enable", None)  # always pop
        ppop(
            "ADetailer Custom mask only top k",
            ["ADetailer Custom mask only top k", "ADetailer Custom method to decide top k masks"],
            cond=0,
        )
        ppop("ADetailer Custom mask min ratio", cond=0.0)
        ppop("ADetailer Custom mask max ratio", cond=1.0)
        ppop("ADetailer Custom x offset", cond=0)
        ppop("ADetailer Custom y offset", cond=0)
        ppop("ADetailer Custom mask merge invert", cond="None")
        ppop("ADetailer Custom mask bbox expansion", cond=0)
        ppop("ADetailer Custom inpaint masked content", cond="original")
        ppop("ADetailer Custom inpaint only masked", ["ADetailer Custom inpaint padding"])
        ppop(
            "ADetailer Custom use inpaint width height",
            [
                "ADetailer Custom use inpaint width height",
                "ADetailer Custom inpaint width",
                "ADetailer Custom inpaint height",
            ],
        )
        ppop(
            "ADetailer Custom use separate steps",
            ["ADetailer Custom use separate steps", "ADetailer Custom steps"],
        )
        ppop(
            "ADetailer Custom use separate CFG scale",
            ["ADetailer Custom use separate CFG scale", "ADetailer Custom CFG scale"],
        )
        ppop(
            "ADetailer Custom use separate checkpoint",
            ["ADetailer Custom use separate checkpoint", "ADetailer Custom checkpoint"],
        )
        ppop(
            "ADetailer Custom use separate VAE",
            ["ADetailer Custom use separate VAE", "ADetailer Custom VAE"],
        )
        ppop(
            "ADetailer Custom use separate sampler",
            [
                "ADetailer Custom use separate sampler",
                "ADetailer Custom sampler",
                "ADetailer Custom scheduler",
            ],
        )
        ppop("ADetailer Custom scheduler", cond="Use same scheduler")
        ppop(
            "ADetailer Custom use separate noise multiplier",
            ["ADetailer Custom use separate noise multiplier", "ADetailer Custom noise multiplier"],
        )

        ppop(
            "ADetailer Custom use separate CLIP skip",
            ["ADetailer Custom use separate CLIP skip", "ADetailer Custom CLIP skip"],
        )

        ppop("ADetailer Custom restore face")
        ppop(
            "ADetailer Custom ControlNet model",
            [
                "ADetailer Custom ControlNet model",
                "ADetailer Custom ControlNet module",
                "ADetailer Custom ControlNet weight",
                "ADetailer Custom ControlNet guidance start",
                "ADetailer Custom ControlNet guidance end",
                "ADetailer Custom ControlNet use ADetailer Custom crop input",
            ],
            cond="None",
        )
        ppop("ADetailer Custom ControlNet module", cond="None")
        ppop("ADetailer Custom ControlNet weight", cond=1.0)
        ppop("ADetailer Custom ControlNet guidance start", cond=0.0)
        ppop("ADetailer Custom ControlNet guidance end", cond=1.0)

        if suffix:
            p = {k + suffix: v for k, v in p.items()}

        return p

    def is_mediapipe(self) -> bool:
        return self.ad_model.lower().startswith("mediapipe")

    def need_skip(self) -> bool:
        return self.ad_model == "None" or self.ad_tab_enable is False


_all_args = [
    ("ad_model", "ADetailer Custom model"),
    ("ad_model_classes", "ADetailer Custom model classes"),
    ("ad_tab_enable", "ADetailer Custom tab enable"),
    ("ad_prompt", "ADetailer Custom prompt"),
    ("ad_prompt_append", "ADetailer Custom prompt append"),
    ("ad_negative_prompt", "ADetailer Custom negative prompt"),
    ("ad_negative_prompt_append", "ADetailer Custom negative prompt append"),
    ("ad_confidence", "ADetailer Custom confidence"),
    ("ad_mask_filter_method", "ADetailer Custom method to decide top k masks"),
    ("ad_mask_k", "ADetailer Custom mask only top k"),
    ("ad_mask_min_ratio", "ADetailer Custom mask min ratio"),
    ("ad_mask_max_ratio", "ADetailer Custom mask max ratio"),
    ("ad_x_offset", "ADetailer Custom x offset"),
    ("ad_y_offset", "ADetailer Custom y offset"),
    ("ad_dilate_erode", "ADetailer Custom dilate erode"),
    ("ad_mask_bbox_expansion", "ADetailer Custom mask bbox expansion"),
    ("ad_mask_merge_invert", "ADetailer Custom mask merge invert"),
    ("ad_mask_blur", "ADetailer Custom mask blur"),
    ("ad_denoising_strength", "ADetailer Custom denoising strength"),
    ("ad_inpaint_masked_content", "ADetailer Custom inpaint masked content"),
    ("ad_inpaint_only_masked", "ADetailer Custom inpaint only masked"),
    ("ad_inpaint_only_masked_padding", "ADetailer Custom inpaint padding"),
    ("ad_use_inpaint_width_height", "ADetailer Custom use inpaint width height"),
    ("ad_inpaint_width", "ADetailer Custom inpaint width"),
    ("ad_inpaint_height", "ADetailer Custom inpaint height"),
    ("ad_use_steps", "ADetailer Custom use separate steps"),
    ("ad_steps", "ADetailer Custom steps"),
    ("ad_use_cfg_scale", "ADetailer Custom use separate CFG scale"),
    ("ad_cfg_scale", "ADetailer Custom CFG scale"),
    ("ad_use_checkpoint", "ADetailer Custom use separate checkpoint"),
    ("ad_checkpoint", "ADetailer Custom checkpoint"),
    ("ad_use_vae", "ADetailer Custom use separate VAE"),
    ("ad_vae", "ADetailer Custom VAE"),
    ("ad_use_sampler", "ADetailer Custom use separate sampler"),
    ("ad_sampler", "ADetailer Custom sampler"),
    ("ad_scheduler", "ADetailer Custom scheduler"),
    ("ad_use_noise_multiplier", "ADetailer Custom use separate noise multiplier"),
    ("ad_noise_multiplier", "ADetailer Custom noise multiplier"),
    ("ad_use_clip_skip", "ADetailer Custom use separate CLIP skip"),
    ("ad_clip_skip", "ADetailer Custom CLIP skip"),
    ("ad_restore_face", "ADetailer Custom restore face"),
    ("ad_controlnet_model", "ADetailer Custom ControlNet model"),
    ("ad_controlnet_module", "ADetailer Custom ControlNet module"),
    ("ad_controlnet_weight", "ADetailer Custom ControlNet weight"),
    ("ad_controlnet_guidance_start", "ADetailer Custom ControlNet guidance start"),
    ("ad_controlnet_guidance_end", "ADetailer Custom ControlNet guidance end"),
    ("ad_controlnet_use_crop_input", "ADetailer Custom ControlNet use ADetailer Custom crop input"),
]

_args = [Arg(*args) for args in _all_args]
ALL_ARGS = ArgsList(_args)

BBOX_SORTBY = [
    "None",
    "Position (left to right)",
    "Position (center to edge)",
    "Area (large to small)",
]

MASK_MERGE_INVERT = ["None", "Merge", "Merge and Invert"]

_script_default = (
    "dynamic_prompting",
    "dynamic_thresholding",
    "wildcard_recursive",
    "wildcards",
    "lora_block_weight",
    "negpip",
)
SCRIPT_DEFAULT = ",".join(sorted(_script_default))

_builtin_script = ("soft_inpainting", "hypertile_script")
BUILTIN_SCRIPT = ",".join(sorted(_builtin_script))


class InpaintBBoxMatchMode(Enum):
    OFF = "Off"
    STRICT = "Strict (SDXL only)"
    FREE = "Free"


INPAINT_BBOX_MATCH_MODES = [
    InpaintBBoxMatchMode.OFF.value,
    InpaintBBoxMatchMode.STRICT.value,
    InpaintBBoxMatchMode.FREE.value,
]
