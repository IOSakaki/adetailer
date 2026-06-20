from __future__ import annotations

import copy
from contextlib import contextmanager

import numpy as np
from lib_controlnet import external_code, global_state
from lib_controlnet.controlnet_ui.controlnet_ui_group import ControlNetUiGroup
from lib_controlnet.external_code import ControlNetUnit

from modules import scripts
from modules.processing import StableDiffusionProcessing

from .common import cn_model_module, cn_model_regex

controlnet_exists = True
controlnet_type = "forge"


def normalize_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def find_script(p: StableDiffusionProcessing, script_title: str) -> scripts.Script:
    script = next((s for s in p.scripts.scripts if s.title() == script_title), None)
    if not script:
        msg = f"Script not found: {script_title!r}"
        raise RuntimeError(msg)
    return script


def add_forge_script_to_adetailer_run(
    p: StableDiffusionProcessing, script_title: str, script_args: list
):
    p.scripts = copy.copy(scripts.scripts_img2img)
    p.scripts.alwayson_scripts = []
    if hasattr(p.scripts, "callback_map"):
        p.scripts.callback_map.clear()
    p.script_args_value = []

    script = copy.copy(find_script(p, script_title))
    script.args_from = len(p.script_args_value)
    script.args_to = len(p.script_args_value) + len(script_args)
    p.scripts.alwayson_scripts.append(script)
    p.script_args_value.extend(script_args)


class ControlNetExt:
    def __init__(self):
        self.cn_available = False
        self.external_cn = external_code

    def init_controlnet(self):
        self.cn_available = True

    @contextmanager
    def disable_batch_dir(self):
        if not hasattr(ControlNetUiGroup, "GLOBAL_CONTROLNET_BATCH_DIR"):
            yield
            return

        batch_dir = ControlNetUiGroup.GLOBAL_CONTROLNET_BATCH_DIR
        ControlNetUiGroup.GLOBAL_CONTROLNET_BATCH_DIR = ""
        try:
            yield
        finally:
            ControlNetUiGroup.GLOBAL_CONTROLNET_BATCH_DIR = batch_dir

    @staticmethod
    def _to_controlnet_image(image):
        return np.asarray(image)

    @staticmethod
    def _to_controlnet_mask(mask):
        if mask is None:
            return None
        if hasattr(mask, "convert"):
            mask = mask.convert("L")
        mask = np.asarray(mask)
        if mask.ndim == 2:
            mask = np.stack([mask] * 3, axis=2)
        elif mask.ndim == 3 and mask.shape[2] == 1:
            mask = np.repeat(mask, 3, axis=2)
        return mask

    @staticmethod
    def _resolve_module_name(module: str | None) -> str | None:
        if module is None:
            return None
        if global_state.get_preprocessor(module) is not None:
            return module

        normalized_module = normalize_name(module)
        if normalized_module == "inpaintnoobai":
            for name in global_state.get_all_preprocessor_names():
                if normalize_name(name).startswith("inpaintnoobai"):
                    return name

        return module

    def update_scripts_args(  # noqa: PLR0913
        self,
        p,
        model: str,
        module: str | None,
        weight: float,
        guidance_start: float,
        guidance_end: float,
        image=None,
        mask=None,
    ):
        if (not self.cn_available) or model == "None":
            return

        if module == "None":
            module = None
        if module is None:
            normalized_model = model.lower().replace("-", "").replace("_", "").replace(" ", "")
            for key, value in cn_model_module.items():
                normalized_key = key.lower().replace("-", "").replace("_", "").replace(" ", "")
                if normalized_key in normalized_model:
                    module = value
                    break
        module = self._resolve_module_name(module)

        p._ad_controlnet_disable_batch_dir = True

        unit_kwargs = {}
        unit_mask = self._to_controlnet_mask(mask)
        if image is not None:
            unit_image = {"image": self._to_controlnet_image(image)}
            if unit_mask is not None:
                unit_image["mask"] = unit_mask
            unit_kwargs["image"] = unit_image

        add_forge_script_to_adetailer_run(
            p,
            "ControlNet",
            [
                ControlNetUnit(
                    enabled=True,
                    model=model,
                    module=module,
                    weight=weight,
                    guidance_start=guidance_start,
                    guidance_end=guidance_end,
                    pixel_perfect=True,
                    **unit_kwargs,
                )
            ],
        )


def get_cn_models() -> list[str]:
    global_state.update_controlnet_filenames()
    models = global_state.get_all_controlnet_names()
    return [m for m in models if m != "None" and cn_model_regex.search(m)]
