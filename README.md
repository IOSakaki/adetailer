# ADetailer Custom

ADetailer Custom is a reForge / Forge Neo oriented fork of
[Bing-su/adetailer](https://github.com/Bing-su/adetailer).

It keeps the basic ADetailer workflow, then adds local fixes and controls for
using ADetailer together with Forge-style ControlNet / LLLite inpaint models.

This is not the official ADetailer repository. Please do not report issues
caused by this fork to upstream ADetailer.

## Target WebUIs

- reForge
- Forge Neo

Other WebUIs may load the extension, but they are not the primary support
target for this fork.

## Main Differences From Upstream ADetailer

- The extension is shown as `ADetailer Custom` in the WebUI.
- Python package names and UI element IDs are separated from upstream ADetailer
  to reduce conflicts when both are installed.
- The displayed ADetailer parameter names in generated image metadata are kept
  as `ADetailer ...` for readability.
- The extension version is reset to `1.0.0` for this custom fork.
- Forge ControlNet model matching includes Anytest / Anima / NoobAI-style
  inpaint model names.
- ADetailer can pass either the ADetailer crop or the full current canvas to
  ControlNet.
- ADetailer can pass an ADetailer-generated mask to Forge ControlNet models
  that require a mask tensor.
- `Prompt append` and `Negative prompt append` can add text to the inherited
  main prompts.
- `Mask bbox expansion` can expand the detected bounding box before masking and
  cropping.
- Japanese localization is included for the ADetailer Custom UI.

## Installation

Manual installation is recommended because the repository name is `adetailer`,
but the intended extension folder name is `ADetailer Custom`.

1. Stop reForge / Forge Neo completely.
2. Open the target WebUI `extensions` folder.
3. Clone this repository into a folder named `ADetailer Custom`.

Windows PowerShell example:

```powershell
cd "G:\Data\Packages\Forge Neo\extensions"
git clone https://github.com/IOSakaki/adetailer.git "ADetailer Custom"
```

reForge example:

```powershell
cd "G:\Data\Packages\reForge\extensions"
git clone https://github.com/IOSakaki/adetailer.git "ADetailer Custom"
```

4. Start reForge / Forge Neo again.

If you install from the WebUI Extensions tab, the folder may be named
`adetailer` because GitHub repository names are used by default. Stop the WebUI
and rename that folder to `ADetailer Custom` if you want the folder name to
match this fork.

## Updating

Stop the WebUI, then run:

```powershell
git -C "path\to\extensions\ADetailer Custom" pull
```

Restart the WebUI after updating.

## Using With Upstream ADetailer

This fork is designed to avoid Python import and UI ID collisions with upstream
ADetailer. You can keep upstream ADetailer installed for comparison.

However, do not enable both ADetailer and ADetailer Custom for the same
generation unless you intentionally want both passes to run. If both extensions
are enabled, both can process the image and the generated image metadata can be
harder to read because this fork intentionally keeps the common `ADetailer ...`
parameter names.

For normal use, enable only `ADetailer Custom`.

## Dependency Safety

WebUI environments share many Python packages, so this installer is conservative.

- In reForge / Forge Neo / webui-like environments, `install.py` installs only
  ADetailer-target packages (`ultralytics`, `mediapipe`, `rich`) with
  `--no-deps` by default.
- Shared packages such as Pillow, numpy, opencv, protobuf, gradio, diffusers,
  and torch are not intentionally upgraded by this extension installer.
- To allow normal dependency resolution explicitly, set
  `ADETAILER_INSTALL_DEPS=1` before launch.

For reForge, launching with `--skip-install` is often the safest choice when
your runtime environment is already working.

If a dependency repair is needed, use the Python executable inside the target
WebUI venv.

Windows example:

```powershell
"path\to\reForge\venv\Scripts\python.exe" -m pip install --force-reinstall "Pillow==10.4.0"
```

## Basic Workflow

1. Enable `ADetailer Custom`.
2. Use the `1st` tab first. Only the first tab is enabled by default.
3. Choose a detector model, such as `face_yolov8n.pt`, `hand_yolov8n.pt`, or a
   segmentation model.
4. Leave the ADetailer prompt blank to inherit the main prompt, or write a local
   prompt for the detected region.
5. Use `Prompt append` / `Negative prompt append` when you want to keep the main
   prompt and only add local instructions.
6. Open `Inpainting` and adjust the inpaint settings.
7. If needed, choose an ADetailer ControlNet model.

## Important Settings

### Inpaint masked content

This controls what is placed under the mask before generation starts.

- `Original image`: keep the original pixels as the starting point.
- `Fill`: fill the masked area with surrounding colors.
- `Latent noise`: start from latent noise.
- `Latent nothing`: treat the masked area as empty latent space.

For strong redraws from scratch, use `Latent nothing` with high denoising.

### Inpaint only masked

On: ADetailer crops around the detected mask, generates inside that crop, then
pastes the result back.

Off: ADetailer processes the full current canvas. This is often better for
Forge LLLite / Anima-style inpaint models when they need to read the whole
image context.

### Mask erosion / dilation

This expands or shrinks the mask shape itself. Use it for small boundary
adjustments.

### Mask bbox expansion

This expands the detected bounding box before the mask and crop are prepared.
It gives ADetailer more room around the detected object.

Use this when the detected region is too tight, for example when redrawing a
hand pose that needs room for extended fingers.

### Use separate width/height

This sets a custom inpaint canvas size for the ADetailer pass. It does not mean
"detect a region of this size"; it changes the canvas size used for the local
ADetailer inpaint operation.

### Use ADetailer crop as ControlNet input

On: ControlNet receives the ADetailer crop.

Off: ControlNet receives the full current canvas together with the ADetailer
mask when possible.

For local face redraws, On is usually a reasonable starting point. For
Anima / LLLite hand or pose changes that need whole-image context, Off is often
the better starting point.

## ControlNet Notes

For Forge ControlNet / LLLite inpaint models:

- Use `None` as the preprocessor unless the model specifically requires a
  preprocessor.
- Anima LLLite inpaint models usually work best with preprocessor `None`.
- NoobAI-style inpaint models should use the matching inpaint preprocessor when
  it is available in the list.
- If a model looks like it is copying the old face or hand too strongly, check
  `Inpaint masked content`, denoising strength, and whether ControlNet is using
  the crop or the full canvas.

## Example Recipes

### Redraw a face with SDXL / NoobAI inpainting

- Detector: face or anime-face segmentation model
- Inpaint masked content: `Latent nothing`
- Denoising strength: high, often `1.0`
- Inpaint only masked: On for local face work
- ControlNet model: NoobAI inpainting model
- ControlNet preprocessor: matching NoobAI inpaint preprocessor if available
- Use ADetailer crop as ControlNet input: On

Use a segmentation detector when you want the mask to follow the face shape
instead of a rectangle.

### Redraw hands with Anima LLLite inpainting

- Detector: hand detector
- Prompt: describe the intended gesture clearly, for example
  `She's waving at me` or `peace sign`
- Inpaint masked content: `Latent nothing`
- Denoising strength: high
- Mask bbox expansion: increase when the hand detection is too tight
- Inpaint only masked: Off is often better when the model needs the full image
  context
- ControlNet model: Anima LLLite inpaint model
- ControlNet preprocessor: `None`
- Use ADetailer crop as ControlNet input: Off as a starting point

## Custom Models

Put Ultralytics YOLO detector models in:

```text
models/adetailer
```

The model name should end with `.pt`.

## License

ADetailer Custom is a derivative of ADetailer and is distributed under the
AGPL-3.0 license, following the upstream project.

Original upstream: https://github.com/Bing-su/adetailer
