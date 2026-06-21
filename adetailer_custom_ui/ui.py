from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import chain
from types import SimpleNamespace
from typing import Any

import gradio as gr

from adetailer_custom import ADETAILER, __version__
from adetailer_custom.args import ALL_ARGS
from adetailer_custom_controlnet_ext import (
    controlnet_exists,
    controlnet_type,
    get_cn_models,
)
from adetailer_custom_ui.conditional import InputAccordion
from modules import shared

if controlnet_type == "forge":
    from lib_controlnet import global_state

    inpaint_modules = list(global_state.get_filtered_preprocessors("Inpaint"))
    noobai_modules = [m for m in inpaint_modules if "noobai" in m.lower()]
    noobai_modules.extend(m for m in inpaint_modules if m not in noobai_modules)

    cn_module_choices = {
        "lllite": ["None"],
        "noobai": noobai_modules,
        "inpaint": inpaint_modules,
        "lineart": list(global_state.get_filtered_preprocessors("Lineart")),
        "openpose": list(global_state.get_filtered_preprocessors("OpenPose")),
        "tile": list(global_state.get_filtered_preprocessors("Tile")),
        "scribble": list(global_state.get_filtered_preprocessors("Scribble")),
        "depth": list(global_state.get_filtered_preprocessors("Depth")),
    }
else:
    cn_module_choices = {
        "lllite": ["None"],
        "noobai": [
            "inpaint_noobai",
            "inpaint_global_harmonious",
            "inpaint_only",
            "inpaint_only+lama",
        ],
        "inpaint": [
            "inpaint_global_harmonious",
            "inpaint_only",
            "inpaint_only+lama",
        ],
        "lineart": [
            "lineart_coarse",
            "lineart_realistic",
            "lineart_anime",
            "lineart_anime_denoise",
        ],
        "openpose": ["openpose_full", "dw_openpose_full"],
        "tile": ["tile_resample", "tile_colorfix", "tile_colorfix+sharp"],
        "scribble": ["t2ia_sketch_pidi"],
        "depth": ["depth_midas", "depth_hand_refiner"],
    }

union = list(chain.from_iterable(cn_module_choices.values()))
cn_module_choices["union"] = union
cn_module_choices["anytest"] = ["None"]

INPAINT_MASKED_CONTENT_CHOICES = (
    ("Original image", "元の画像", "original"),
    ("Fill", "埋める", "fill"),
    ("Latent noise", "潜在空間でのノイズ", "latent noise"),
    ("Latent nothing", "潜在空間における無", "latent nothing"),
)


def is_japanese_localization() -> bool:
    locale = str(getattr(shared.opts, "localization", "") or "").lower()
    return locale.startswith("ja") or locale in {"ks_jp"}


def localized_choices(choices: tuple[tuple[str, str, str], ...]):
    label_index = 1 if is_japanese_localization() else 0
    return [(choice[label_index], choice[2]) for choice in choices]


MASKED_CONTENT_CHOICES = localized_choices(INPAINT_MASKED_CONTENT_CHOICES)

MASK_MERGE_INVERT_CHOICES = [
    ("None", "None"),
    ("Merge masks", "Merge"),
    ("Merge and invert masks", "Merge and Invert"),
]


class Widgets(SimpleNamespace):
    def tolist(self):
        return [getattr(self, attr) for attr in ALL_ARGS.attrs]


@dataclass
class WebuiInfo:
    ad_model_list: list[str]
    sampler_names: list[str]
    scheduler_names: list[str]
    t2i_button: gr.Button
    i2i_button: gr.Button
    checkpoints_list: list[str]
    vae_list: list[str]


def gr_interactive(value: bool = True):
    return gr.update(interactive=value)


def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


def suffix(n: int, c: str = " ") -> str:
    return "" if n == 0 else c + ordinal(n + 1)


def on_widget_change(state: dict, value: Any, *, attr: str):
    if "is_api" in state:
        state = state.copy()
        state.pop("is_api")
    state[attr] = value
    return state


def on_generate_click(state: dict, *values: Any):
    for attr, value in zip(ALL_ARGS.attrs, values):
        state[attr] = value  # noqa: PERF403
    state["is_api"] = ()
    return state


def on_ad_model_update(model: str):
    if "-world" in model:
        return gr.update(
            visible=True,
            placeholder="Comma separated class names to detect, ex: 'person,cat'. default: COCO 80 classes",
        )
    return gr.update(visible=False, placeholder="")


def normalize_cn_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def on_cn_model_update(cn_model_name: str):
    cn_model_name = cn_model_name.replace("inpaint_depth", "depth")
    normalized_model_name = normalize_cn_name(cn_model_name)
    for t in cn_module_choices:
        if normalize_cn_name(t) in normalized_model_name:
            choices = cn_module_choices[t]
            return gr.update(visible=True, choices=choices, value=choices[0])
    return gr.update(visible=False, choices=["None"], value="None")


def elem_id(item_id: str, n: int, is_img2img: bool) -> str:
    tab = "img2img" if is_img2img else "txt2img"
    suf = suffix(n, "_")
    return f"script_{tab}_adetailer_custom_{item_id}{suf}"


def state_init(w: Widgets) -> dict[str, Any]:
    return {attr: getattr(w, attr).value for attr in ALL_ARGS.attrs}


def adui(
    num_models: int,
    is_img2img: bool,
    webui_info: WebuiInfo,
):
    states = []
    infotext_fields = []
    eid = partial(elem_id, n=0, is_img2img=is_img2img)

    with InputAccordion(
        value=False,
        elem_id=eid("ad_main_accordion"),
        label=ADETAILER,
        visible=True,
    ) as ad_enable:
        with gr.Row():
            with gr.Column(scale=8):
                ad_skip_img2img = gr.Checkbox(
                    label="Skip img2img",
                    tooltip="Skip the main img2img pass and run only ADetailer. Useful when you only want detected areas to be edited.",
                    value=False,
                    visible=is_img2img,
                    elem_id=eid("ad_skip_img2img"),
                )

            with gr.Column(scale=1, min_width=180):
                gr.Markdown(
                    f"v{__version__}",
                    elem_id=eid("ad_version"),
                )

        infotext_fields.append((ad_enable, "ADetailer enable"))
        infotext_fields.append((ad_skip_img2img, "ADetailer skip img2img"))

        with gr.Group(), gr.Tabs():
            for n in range(num_models):
                with gr.Tab(ordinal(n + 1)):
                    state, infofields = one_ui_group(
                        n=n,
                        is_img2img=is_img2img,
                        webui_info=webui_info,
                    )

                states.append(state)
                infotext_fields.extend(infofields)

    # components: [bool, bool, dict, dict, ...]
    components = [ad_enable, ad_skip_img2img, *states]
    return components, infotext_fields


def one_ui_group(n: int, is_img2img: bool, webui_info: WebuiInfo):
    w = Widgets()
    eid = partial(elem_id, n=n, is_img2img=is_img2img)

    model_choices = (
        [*webui_info.ad_model_list, "None"]
        if n == 0
        else ["None", *webui_info.ad_model_list]
    )

    with gr.Group():
        with gr.Row(variant="compact"):
            w.ad_tab_enable = gr.Checkbox(
                label="Enable this tab",
                tooltip="Turn this ADetailer tab on or off.",
                value=n == 0,
                visible=True,
                elem_id=eid("ad_tab_enable"),
            )
            w.ad_tab_enable.do_not_save_to_config = True

        with gr.Row():
            w.ad_model = gr.Dropdown(
                label="ADetailer detector",
                choices=model_choices,
                value=model_choices[0],
                visible=True,
                type="value",
                elem_id=eid("ad_model"),
                tooltip="Select the detector model that finds faces, hands, or other regions to edit.",
            )

        with gr.Row():
            w.ad_model_classes = gr.Textbox(
                label="ADetailer detector classes",
                tooltip="For YOLO World models, enter class names to detect, separated by commas.",
                value="",
                visible=False,
                elem_id=eid("ad_model_classes"),
            )

            w.ad_model.change(
                on_ad_model_update,
                inputs=w.ad_model,
                outputs=w.ad_model_classes,
                queue=False,
            )

    gr.HTML("<br>")

    with gr.Group():
        with gr.Row(elem_id=eid("ad_toprow_prompt")):
            w.ad_prompt = gr.Textbox(
                value="",
                label="ad_prompt",
                show_label=False,
                lines=3,
                placeholder="ADetailer prompt\nIf blank, the main prompt is used.",
                elem_id=eid("ad_prompt"),
            )

        with gr.Row(elem_id=eid("ad_toprow_negative_prompt")):
            w.ad_negative_prompt = gr.Textbox(
                value="",
                label="ad_negative_prompt",
                show_label=False,
                lines=2,
                placeholder="ADetailer negative prompt\nIf blank, the main negative prompt is used.",
                elem_id=eid("ad_negative_prompt"),
            )

        with gr.Row(elem_id=eid("ad_toprow_prompt_append")):
            w.ad_prompt_append = gr.Textbox(
                value="",
                label="ad_prompt_append",
                show_label=False,
                lines=2,
                placeholder="Prompt append\nAppended to the final ADetailer prompt.",
                elem_id=eid("ad_prompt_append"),
            )

        with gr.Row(elem_id=eid("ad_toprow_negative_prompt_append")):
            w.ad_negative_prompt_append = gr.Textbox(
                value="",
                label="ad_negative_prompt_append",
                show_label=False,
                lines=2,
                placeholder="Negative prompt append\nAppended to the final ADetailer negative prompt.",
                elem_id=eid("ad_negative_prompt_append"),
            )

    with gr.Group():
        with gr.Accordion(
            "Detection", open=False, elem_id=eid("ad_detection_accordion")
        ):
            detection(w, n, is_img2img)

        with gr.Accordion(
            "Mask Preprocessing",
            open=False,
            elem_id=eid("ad_mask_preprocessing_accordion"),
        ):
            mask_preprocessing(w, n, is_img2img)

        with gr.Accordion(
            "Inpainting", open=False, elem_id=eid("ad_inpainting_accordion")
        ):
            inpainting(w, n, is_img2img, webui_info)

    with gr.Group():
        controlnet(w, n, is_img2img)

    state = gr.State(lambda: state_init(w))

    for attr in ALL_ARGS.attrs:
        widget = getattr(w, attr)
        on_change = partial(on_widget_change, attr=attr)
        widget.change(fn=on_change, inputs=[state, widget], outputs=state, queue=False)

    all_inputs = [state, *w.tolist()]
    target_button = webui_info.i2i_button if is_img2img else webui_info.t2i_button
    target_button.click(
        fn=on_generate_click, inputs=all_inputs, outputs=state, queue=False
    )

    infotext_fields = [(getattr(w, attr), name + suffix(n)) for attr, name in ALL_ARGS]

    return state, infotext_fields


def detection(w: Widgets, n: int, is_img2img: bool):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)

    with gr.Row():
        with gr.Column(variant="compact"):
            w.ad_confidence = gr.Slider(
                label="Detection confidence",
                tooltip="Higher values keep only more confident detections. Lower values find more regions, but may include mistakes.",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.3,
                visible=True,
                elem_id=eid("ad_confidence"),
            )
            w.ad_mask_filter_method = gr.Radio(
                choices=["Area", "Confidence"],
                value="Area",
                label="Top-k filter method",
                tooltip="Choose whether top detections are selected by mask area or detection confidence.",
                visible=True,
                elem_id=eid("ad_mask_filter_method"),
            )
            w.ad_mask_k = gr.Slider(
                label="Mask only top k",
                tooltip="0 keeps all detections. Use 1 to edit only the highest-ranked detected region.",
                minimum=0,
                maximum=10,
                step=1,
                value=0,
                visible=True,
                elem_id=eid("ad_mask_k"),
            )

        with gr.Column(variant="compact"):
            w.ad_mask_min_ratio = gr.Slider(
                label="Mask min area ratio",
                tooltip="Ignore detected regions smaller than this fraction of the whole image.",
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                value=0.0,
                visible=True,
                elem_id=eid("ad_mask_min_ratio"),
            )
            w.ad_mask_max_ratio = gr.Slider(
                label="Mask max area ratio",
                tooltip="Ignore detected regions larger than this fraction of the whole image.",
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                value=1.0,
                visible=True,
                elem_id=eid("ad_mask_max_ratio"),
            )


def mask_preprocessing(w: Widgets, n: int, is_img2img: bool):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)

    with gr.Group():
        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_x_offset = gr.Slider(
                    label="Mask x offset",
                    tooltip="Move the mask horizontally. Positive values move it right; negative values move it left.",
                    minimum=-200,
                    maximum=200,
                    step=1,
                    value=0,
                    visible=True,
                    elem_id=eid("ad_x_offset"),
                )
                w.ad_y_offset = gr.Slider(
                    label="Mask y offset",
                    tooltip="Move the mask vertically. Positive values move it up; negative values move it down.",
                    minimum=-200,
                    maximum=200,
                    step=1,
                    value=0,
                    visible=True,
                    elem_id=eid("ad_y_offset"),
                )

            with gr.Column(variant="compact"):
                w.ad_dilate_erode = gr.Slider(
                    label="Mask erosion / dilation",
                    tooltip="Negative values shrink the mask. Positive values expand the mask.",
                    minimum=-128,
                    maximum=128,
                    step=4,
                    value=4,
                    visible=True,
                    elem_id=eid("ad_dilate_erode"),
                )
                w.ad_mask_bbox_expansion = gr.Slider(
                    label="Mask bbox expansion",
                    info="Expand the detected bounding box before making the mask. Useful when the redraw needs more room.",
                    minimum=0,
                    maximum=512,
                    step=4,
                    value=0,
                    visible=True,
                    elem_id=eid("ad_mask_bbox_expansion"),
                )

        with gr.Row():
            w.ad_mask_merge_invert = gr.Radio(
                label="Mask merge mode",
                choices=MASK_MERGE_INVERT_CHOICES,
                value="None",
                elem_id=eid("ad_mask_merge_invert"),
                tooltip="None keeps masks separate. Merge combines all masks. Merge and invert edits everything outside the combined mask.",
            )


def inpainting(w: Widgets, n: int, is_img2img: bool, webui_info: WebuiInfo):  # noqa: PLR0915
    eid = partial(elem_id, n=n, is_img2img=is_img2img)

    with gr.Group():
        with gr.Row():
            w.ad_mask_blur = gr.Slider(
                label="Inpaint mask blur",
                tooltip="Soften the edge of the inpaint mask so the edited area blends into the image.",
                minimum=0,
                maximum=64,
                step=1,
                value=4,
                visible=True,
                elem_id=eid("ad_mask_blur"),
            )

            w.ad_denoising_strength = gr.Slider(
                label="Inpaint denoising strength",
                info="How strongly to redraw the masked area. 1.0 changes it the most; lower values preserve more of the original.",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.4,
                visible=True,
                elem_id=eid("ad_denoising_strength"),
            )

        with gr.Row():
            w.ad_inpaint_masked_content = gr.Dropdown(
                label="Inpaint masked content",
                info="Choose what is placed under the mask before generation starts.",
                choices=MASKED_CONTENT_CHOICES,
                value="original",
                visible=True,
                type="value",
                elem_id=eid("ad_inpaint_masked_content"),
            )

        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_inpaint_only_masked = gr.Checkbox(
                    label="Inpaint only masked",
                    info="Crop around the mask, generate there, then paste it back. Turn off to process the whole canvas.",
                    value=True,
                    visible=True,
                    elem_id=eid("ad_inpaint_only_masked"),
                )
                w.ad_inpaint_only_masked_padding = gr.Slider(
                    label="Inpaint only masked padding",
                    tooltip="Extra pixels added around the mask crop when Inpaint only masked is on.",
                    minimum=0,
                    maximum=256,
                    step=4,
                    value=32,
                    visible=True,
                    elem_id=eid("ad_inpaint_only_masked_padding"),
                )

                w.ad_inpaint_only_masked.change(
                    gr_interactive,
                    inputs=w.ad_inpaint_only_masked,
                    outputs=w.ad_inpaint_only_masked_padding,
                    queue=False,
                )

            with gr.Column(variant="compact"):
                w.ad_use_inpaint_width_height = gr.Checkbox(
                    label="Use separate width/height",
                    info="Use a custom inpaint canvas size for ADetailer instead of the main image size.",
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_inpaint_width_height"),
                )

                w.ad_inpaint_width = gr.Slider(
                    label="Inpaint width",
                    tooltip="Width of the ADetailer inpaint canvas when separate width/height is enabled.",
                    minimum=64,
                    maximum=2048,
                    step=4,
                    value=512,
                    visible=True,
                    elem_id=eid("ad_inpaint_width"),
                )

                w.ad_inpaint_height = gr.Slider(
                    label="Inpaint height",
                    tooltip="Height of the ADetailer inpaint canvas when separate width/height is enabled.",
                    minimum=64,
                    maximum=2048,
                    step=4,
                    value=512,
                    visible=True,
                    elem_id=eid("ad_inpaint_height"),
                )

                w.ad_use_inpaint_width_height.change(
                    lambda value: (gr_interactive(value), gr_interactive(value)),
                    inputs=w.ad_use_inpaint_width_height,
                    outputs=[w.ad_inpaint_width, w.ad_inpaint_height],
                    queue=False,
                )

        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_use_steps = gr.Checkbox(
                    label="Use separate steps",
                    tooltip="Use a different sampling step count for the ADetailer pass.",
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_steps"),
                )

                w.ad_steps = gr.Slider(
                    label="ADetailer steps",
                    tooltip="Sampling steps used only by ADetailer when separate steps is enabled.",
                    minimum=1,
                    maximum=150,
                    step=1,
                    value=28,
                    visible=True,
                    elem_id=eid("ad_steps"),
                )

                w.ad_use_steps.change(
                    gr_interactive,
                    inputs=w.ad_use_steps,
                    outputs=w.ad_steps,
                    queue=False,
                )

            with gr.Column(variant="compact"):
                w.ad_use_cfg_scale = gr.Checkbox(
                    label="Use separate CFG scale",
                    tooltip="Use a different prompt guidance strength for the ADetailer pass.",
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_cfg_scale"),
                )

                w.ad_cfg_scale = gr.Slider(
                    label="ADetailer CFG scale",
                    tooltip="Prompt guidance strength used only by ADetailer when separate CFG scale is enabled.",
                    minimum=0.0,
                    maximum=30.0,
                    step=0.5,
                    value=7.0,
                    visible=True,
                    elem_id=eid("ad_cfg_scale"),
                )

                w.ad_use_cfg_scale.change(
                    gr_interactive,
                    inputs=w.ad_use_cfg_scale,
                    outputs=w.ad_cfg_scale,
                    queue=False,
                )

        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_use_checkpoint = gr.Checkbox(
                    label="Use separate checkpoint",
                    tooltip="Use a different checkpoint only for the ADetailer pass.",
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_checkpoint"),
                )

                ckpts = ["Use same checkpoint", *webui_info.checkpoints_list]

                w.ad_checkpoint = gr.Dropdown(
                    label="ADetailer checkpoint",
                    tooltip="Checkpoint used only by ADetailer when separate checkpoint is enabled.",
                    choices=ckpts,
                    value=ckpts[0],
                    visible=True,
                    elem_id=eid("ad_checkpoint"),
                )

            with gr.Column(variant="compact"):
                w.ad_use_vae = gr.Checkbox(
                    label="Use separate VAE",
                    tooltip="Use a different VAE only for the ADetailer pass.",
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_vae"),
                )

                vaes = ["Use same VAE", *webui_info.vae_list]

                w.ad_vae = gr.Dropdown(
                    label="ADetailer VAE",
                    tooltip="VAE used only by ADetailer when separate VAE is enabled.",
                    choices=vaes,
                    value=vaes[0],
                    visible=True,
                    elem_id=eid("ad_vae"),
                )

        with gr.Row(), gr.Column(variant="compact"):
            w.ad_use_sampler = gr.Checkbox(
                label="Use separate sampler",
                tooltip="Use a different sampler and scheduler only for the ADetailer pass.",
                value=False,
                visible=True,
                elem_id=eid("ad_use_sampler"),
            )

            sampler_names = [
                "Use same sampler",
                *webui_info.sampler_names,
            ]

            with gr.Row():
                w.ad_sampler = gr.Dropdown(
                    label="ADetailer sampler",
                    tooltip="Sampler used only by ADetailer when separate sampler is enabled.",
                    choices=sampler_names,
                    value=sampler_names[1],
                    visible=True,
                    elem_id=eid("ad_sampler"),
                )

                scheduler_names = [
                    "Use same scheduler",
                    *webui_info.scheduler_names,
                ]
                w.ad_scheduler = gr.Dropdown(
                    label="ADetailer scheduler",
                    tooltip="Scheduler used only by ADetailer when separate sampler is enabled.",
                    choices=scheduler_names,
                    value=scheduler_names[0],
                    visible=len(scheduler_names) > 1,
                    elem_id=eid("ad_scheduler"),
                )

                w.ad_use_sampler.change(
                    lambda value: (gr_interactive(value), gr_interactive(value)),
                    inputs=w.ad_use_sampler,
                    outputs=[w.ad_sampler, w.ad_scheduler],
                    queue=False,
                )

        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_use_noise_multiplier = gr.Checkbox(
                    label="Use separate noise multiplier",
                    tooltip="Use a different img2img noise multiplier only for the ADetailer pass.",
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_noise_multiplier"),
                )

                w.ad_noise_multiplier = gr.Slider(
                    label="Noise multiplier for img2img",
                    tooltip="Controls the amount of initial noise in the ADetailer img2img pass.",
                    minimum=0.5,
                    maximum=1.5,
                    step=0.01,
                    value=1.0,
                    visible=True,
                    elem_id=eid("ad_noise_multiplier"),
                )

                w.ad_use_noise_multiplier.change(
                    gr_interactive,
                    inputs=w.ad_use_noise_multiplier,
                    outputs=w.ad_noise_multiplier,
                    queue=False,
                )

            with gr.Column(variant="compact"):
                w.ad_use_clip_skip = gr.Checkbox(
                    label="Use separate CLIP skip",
                    tooltip="Use a different CLIP skip value only for the ADetailer pass.",
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_clip_skip"),
                )

                w.ad_clip_skip = gr.Slider(
                    label="ADetailer CLIP skip",
                    tooltip="CLIP skip value used only by ADetailer when separate CLIP skip is enabled.",
                    minimum=1,
                    maximum=12,
                    step=1,
                    value=1,
                    visible=True,
                    elem_id=eid("ad_clip_skip"),
                )

                w.ad_use_clip_skip.change(
                    gr_interactive,
                    inputs=w.ad_use_clip_skip,
                    outputs=w.ad_clip_skip,
                    queue=False,
                )

        with gr.Row(), gr.Column(variant="compact"):
            w.ad_restore_face = gr.Checkbox(
                label="Restore faces after ADetailer",
                tooltip="Apply face restoration after the ADetailer pass.",
                value=False,
                elem_id=eid("ad_restore_face"),
            )


def controlnet(w: Widgets, n: int, is_img2img: bool):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)
    cn_models = ["None", "Passthrough", *get_cn_models()]

    with gr.Row(variant="panel"):
        with gr.Column(variant="compact"):
            w.ad_controlnet_model = gr.Dropdown(
                label="ControlNet model",
                tooltip="ControlNet or LLLite model used only during the ADetailer pass. Passthrough keeps existing ControlNet units instead of adding a new ADetailer unit.",
                choices=cn_models,
                value="None",
                visible=True,
                type="value",
                interactive=controlnet_exists,
                elem_id=eid("ad_controlnet_model"),
            )

            w.ad_controlnet_module = gr.Dropdown(
                label="ControlNet preprocessor",
                info="Preprocessor used before ControlNet. For Anima LLLite inpaint models, None is usually the correct choice.",
                choices=["None"],
                value="None",
                visible=False,
                type="value",
                interactive=controlnet_exists,
                elem_id=eid("ad_controlnet_module"),
            )

            w.ad_controlnet_weight = gr.Slider(
                label="ControlNet weight",
                tooltip="Strength of the ControlNet effect during the ADetailer pass.",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=1.0,
                visible=True,
                interactive=controlnet_exists,
                elem_id=eid("ad_controlnet_weight"),
            )

            w.ad_controlnet_model.change(
                on_cn_model_update,
                inputs=w.ad_controlnet_model,
                outputs=w.ad_controlnet_module,
                queue=False,
            )

        with gr.Column(variant="compact"):
            w.ad_controlnet_guidance_start = gr.Slider(
                label="ControlNet guidance start",
                tooltip="Generation progress where ControlNet starts to affect the image. 0.0 means from the beginning.",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.0,
                visible=True,
                interactive=controlnet_exists,
                elem_id=eid("ad_controlnet_guidance_start"),
            )

            w.ad_controlnet_guidance_end = gr.Slider(
                label="ControlNet guidance end",
                tooltip="Generation progress where ControlNet stops affecting the image. 1.0 means until the end.",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=1.0,
                visible=True,
                interactive=controlnet_exists,
                elem_id=eid("ad_controlnet_guidance_end"),
            )

            w.ad_controlnet_use_crop_input = gr.Checkbox(
                label="Use ADetailer crop as ControlNet input",
                info="On uses the ADetailer crop as ControlNet input. Off uses the full current canvas with the ADetailer mask.",
                value=True,
                visible=True,
                interactive=controlnet_exists,
                elem_id=eid("ad_controlnet_use_crop_input"),
            )
