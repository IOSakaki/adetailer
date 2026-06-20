# ADetailer Custom

## 日本語

### 概要

ADetailer Customは、
[Bing-su/adetailer](https://github.com/Bing-su/adetailer) をもとにした、
reForge / Forge Neo向けの非公式フォークです。

本家ADetailerの基本的な流れを保ちつつ、Forge系ControlNet /
LLLiteインペイントモデルと組み合わせて使うための調整やUI項目を追加しています。

このリポジトリは本家ADetailerではありません。また、本家ADetailerの作者・貢献者による公式版、または公認版ではありません。
このフォーク固有の不具合を、本家ADetailer側へ報告しないでください。

ADetailer Customは、本家ADetailerの開発と公開があって初めて成立している派生版です。
本家作者・貢献者の皆様に感謝し、上流プロジェクトの著作権表示、ライセンス、開発方針を尊重します。

### 対象WebUI

- reForge
- Forge Neo

その他のWebUIでも読み込める可能性はありますが、このフォークの主なサポート対象ではありません。

### 本家ADetailerとの主な違い

- WebUI上では `ADetailer Custom` と表示されます。
- Pythonパッケージ名とUI要素IDを本家ADetailerから分離し、同時インストール時の衝突を減らしています。
- 生成画像メタデータ内のADetailerパラメータ名は、読みやすさのため `ADetailer ...` の表記を維持しています。
- このカスタムフォークのバージョンは `1.0.0` から始めています。
- Forge ControlNetのモデル検索で、Anytest / Anima / NoobAI系インペイントモデル名を拾いやすくしています。
- ControlNetへ渡す入力を、ADetailerの切り出し画像または現在のキャンバス全体から選べます。
- マスクが必要なForge ControlNetモデルへ、ADetailerで作ったマスクを渡せるようにしています。
- `Prompt append` と `Negative prompt append` で、メインプロンプトを継承しつつ末尾へ追記できます。
- `Mask bbox expansion` で、検出された四角い範囲をマスク作成・切り出し前に広げられます。
- ADetailer Custom UI向けの日本語ローカライズを含みます。

### インストール

このリポジトリ名は `adetailer` ですが、意図している拡張フォルダ名は `ADetailer Custom` です。
そのため、手動インストールを推奨します。

1. reForge / Forge Neoを完全に終了します。
2. 対象WebUIの `extensions` フォルダを開きます。
3. このリポジトリを `ADetailer Custom` というフォルダ名でcloneします。

Forge NeoのWindows PowerShell例:

```powershell
cd "G:\Data\Packages\Forge Neo\extensions"
git clone https://github.com/IOSakaki/adetailer.git "ADetailer Custom"
```

reForgeの例:

```powershell
cd "G:\Data\Packages\reForge\extensions"
git clone https://github.com/IOSakaki/adetailer.git "ADetailer Custom"
```

4. reForge / Forge Neoを起動します。

WebUIのExtensionsタブからインストールした場合、GitHubのリポジトリ名に合わせてフォルダ名が `adetailer` になることがあります。
フォルダ名もこのフォークに合わせたい場合は、WebUIを停止してから `ADetailer Custom` にリネームしてください。

### 更新

WebUIを停止してから、次を実行します。

```powershell
git -C "path\to\extensions\ADetailer Custom" pull
```

更新後、WebUIを再起動してください。

### 本家ADetailerとの同時利用

このフォークは、本家ADetailerとのPython importやUI IDの衝突を避けるように作っています。
比較用として本家ADetailerを同時にインストールしておくことは可能です。

ただし、同じ生成で本家ADetailerとADetailer Customを両方有効にすると、両方の処理が走ります。
意図して二重に処理したい場合を除き、通常は `ADetailer Custom` だけを有効にしてください。

また、このフォークは生成画像メタデータの読みやすさを優先して、共通の `ADetailer ...` パラメータ名を維持しています。
両方を同時に使うと、メタデータの読み分けが難しくなる場合があります。

### 依存関係の安全性

WebUI環境では多くのPythonパッケージを共有しているため、このインストーラーは保守的に動作します。

- reForge / Forge Neo / WebUI系の環境では、`install.py` はADetailer向けパッケージ
  (`ultralytics`, `mediapipe`, `rich`) だけを、標準では `--no-deps` 付きでインストールします。
- Pillow, numpy, opencv, protobuf, gradio, diffusers, torch などの共有パッケージは、
  この拡張のインストーラーから意図的にアップグレードしません。
- 通常の依存解決を明示的に許可したい場合は、起動前に `ADETAILER_INSTALL_DEPS=1` を設定してください。

reForgeでは、すでに動いている環境を壊さないために `--skip-install` 付きで起動するのが安全な場合があります。

依存関係を修復する必要がある場合は、対象WebUIのvenv内にあるPythonを使ってください。

Windows例:

```powershell
"path\to\reForge\venv\Scripts\python.exe" -m pip install --force-reinstall "Pillow==10.4.0"
```

### 基本的な使い方

1. `ADetailer Custom` を有効にします。
2. まず `1st` タブを使います。初期状態では1つ目のタブだけが有効です。
3. `face_yolov8n.pt`, `hand_yolov8n.pt`, セグメンテーションモデルなど、検出モデルを選びます。
4. ADetailer用プロンプトを空欄にすると、メインプロンプトを継承します。局所的に変えたい場合は、ADetailer用プロンプトを書きます。
5. メインプロンプトを維持したまま局所指示だけ足したい場合は、`Prompt append` / `Negative prompt append` を使います。
6. `Inpainting` を開いて、インペイント設定を調整します。
7. 必要に応じて、ADetailer用ControlNetモデルを選びます。

### 重要な設定

#### Inpaint masked content

生成を始める前に、マスクされた範囲をどの状態として扱うかを決めます。

- `Original image`: 元画像のピクセルを開始点として残します。
- `Fill`: マスク部分を周辺色で埋めます。
- `Latent noise`: 潜在空間のノイズから始めます。
- `Latent nothing`: マスク部分を空の潜在空間として扱います。

ゼロから強く描き直したい場合は、高いdenoising strengthと `Latent nothing` を組み合わせます。

#### Inpaint only masked

ON: 検出マスクの周辺を切り出して、その範囲だけを生成し、結果を元画像に貼り戻します。

OFF: 現在のキャンバス全体を処理します。Forge LLLite / Anima系インペイントモデルが画像全体の文脈を読む必要がある場合は、OFFのほうが向いていることがあります。

#### Mask erosion / dilation

マスク形状そのものを広げたり縮めたりします。境界の小さな調整に使います。

#### Mask bbox expansion

マスク作成と切り出しの前に、検出された四角い範囲を広げます。
ADetailerが検出物の周囲を見る余白を増やします。

手のポーズ変更など、検出範囲が狭すぎて伸ばした指を描く余地がない場合に使います。

#### Use separate width/height

ADetailer処理で使うインペイントキャンバスサイズを指定します。
「このサイズの範囲を検出する」という意味ではなく、局所インペイントに使うキャンバスサイズを変える設定です。

#### Use ADetailer crop as ControlNet input

ON: ControlNetへADetailerの切り出し画像を渡します。

OFF: 可能な場合、ADetailerマスクと一緒に現在のキャンバス全体をControlNetへ渡します。

顔の局所修正ではONが無難な出発点です。
Anima / LLLiteで手やポーズを変える場合のように、画像全体の文脈が必要なときはOFFから試すのが向いています。

### ControlNetについて

Forge ControlNet / LLLiteインペイントモデルでは、次を目安にしてください。

- モデルが特定のプリプロセッサを要求しない限り、プリプロセッサは `None` を使います。
- Anima LLLite系インペイントモデルは、多くの場合 `None` が適しています。
- NoobAI系インペイントモデルでは、一覧に対応するinpaintプリプロセッサがある場合、それを使います。
- 古い顔や手を強くコピーしてしまう場合は、`Inpaint masked content`、denoising strength、ControlNet入力が切り出し画像か全体キャンバスかを確認してください。

### 使用例

#### SDXL / NoobAIインペイントで顔を描き直す

- Detector: 顔検出モデル、またはアニメ顔向けセグメンテーションモデル
- Inpaint masked content: `Latent nothing`
- Denoising strength: 高め、多くの場合 `1.0`
- Inpaint only masked: 顔の局所修正ではON
- ControlNet model: NoobAI系インペイントモデル
- ControlNet preprocessor: 対応するNoobAI inpaintプリプロセッサがある場合はそれを選ぶ
- Use ADetailer crop as ControlNet input: ON

顔の形に沿ったマスクにしたい場合は、四角いbbox検出ではなくセグメンテーションモデルを使ってください。

#### Anima LLLiteインペイントで手を描き直す

- Detector: 手検出モデル
- Prompt: `She's waving at me` や `peace sign` など、意図するジェスチャーを明確に書く
- Inpaint masked content: `Latent nothing`
- Denoising strength: 高め
- Mask bbox expansion: 手の検出範囲が狭い場合に増やす
- Inpaint only masked: モデルが画像全体の文脈を必要とする場合はOFFが向いていることが多い
- ControlNet model: Anima LLLite系インペイントモデル
- ControlNet preprocessor: `None`
- Use ADetailer crop as ControlNet input: まずOFFから試す

### カスタム検出モデル

Ultralytics YOLO検出モデルは次の場所に置きます。

```text
models/adetailer
```

モデル名は `.pt` で終わる必要があります。

### 謝辞とライセンス

ADetailer Customは、本家ADetailerの派生物です。
本家ADetailerおよび関連するオープンソースプロジェクトの作者・貢献者に感謝します。

このフォークは、本家プロジェクトと同じくAGPL-3.0ライセンスのもとで配布します。
上流由来部分の著作権表示とライセンスは、それぞれの権利者に帰属します。
このフォークは、本リポジトリ内で加えた変更についてのみ責任を持ちます。

本家ADetailer: https://github.com/Bing-su/adetailer

---

## English

### Overview

ADetailer Custom is a non-official reForge / Forge Neo oriented fork of
[Bing-su/adetailer](https://github.com/Bing-su/adetailer).

It keeps the basic ADetailer workflow, then adds local fixes and controls for
using ADetailer together with Forge-style ControlNet / LLLite inpaint models.

This is not the official ADetailer repository, and it is not endorsed by the
original ADetailer authors or contributors. Please do not report issues caused
by this fork to upstream ADetailer.

ADetailer Custom exists because the original ADetailer project was developed
and shared publicly. This fork respects the upstream project's copyright
notices, license, and development boundaries, with thanks to the original
authors and contributors.

### Target WebUIs

- reForge
- Forge Neo

Other WebUIs may load the extension, but they are not the primary support
target for this fork.

### Main Differences From Upstream ADetailer

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

### Installation

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

### Updating

Stop the WebUI, then run:

```powershell
git -C "path\to\extensions\ADetailer Custom" pull
```

Restart the WebUI after updating.

### Using With Upstream ADetailer

This fork is designed to avoid Python import and UI ID collisions with upstream
ADetailer. You can keep upstream ADetailer installed for comparison.

However, do not enable both ADetailer and ADetailer Custom for the same
generation unless you intentionally want both passes to run. If both extensions
are enabled, both can process the image and the generated image metadata can be
harder to read because this fork intentionally keeps the common `ADetailer ...`
parameter names.

For normal use, enable only `ADetailer Custom`.

### Dependency Safety

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

### Basic Workflow

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

### Important Settings

#### Inpaint masked content

This controls what is placed under the mask before generation starts.

- `Original image`: keep the original pixels as the starting point.
- `Fill`: fill the masked area with surrounding colors.
- `Latent noise`: start from latent noise.
- `Latent nothing`: treat the masked area as empty latent space.

For strong redraws from scratch, use `Latent nothing` with high denoising.

#### Inpaint only masked

On: ADetailer crops around the detected mask, generates inside that crop, then
pastes the result back.

Off: ADetailer processes the full current canvas. This is often better for
Forge LLLite / Anima-style inpaint models when they need to read the whole
image context.

#### Mask erosion / dilation

This expands or shrinks the mask shape itself. Use it for small boundary
adjustments.

#### Mask bbox expansion

This expands the detected bounding box before the mask and crop are prepared.
It gives ADetailer more room around the detected object.

Use this when the detected region is too tight, for example when redrawing a
hand pose that needs room for extended fingers.

#### Use separate width/height

This sets a custom inpaint canvas size for the ADetailer pass. It does not mean
"detect a region of this size"; it changes the canvas size used for the local
ADetailer inpaint operation.

#### Use ADetailer crop as ControlNet input

On: ControlNet receives the ADetailer crop.

Off: ControlNet receives the full current canvas together with the ADetailer
mask when possible.

For local face redraws, On is usually a reasonable starting point. For
Anima / LLLite hand or pose changes that need whole-image context, Off is often
the better starting point.

### ControlNet Notes

For Forge ControlNet / LLLite inpaint models:

- Use `None` as the preprocessor unless the model specifically requires a
  preprocessor.
- Anima LLLite inpaint models usually work best with preprocessor `None`.
- NoobAI-style inpaint models should use the matching inpaint preprocessor when
  it is available in the list.
- If a model looks like it is copying the old face or hand too strongly, check
  `Inpaint masked content`, denoising strength, and whether ControlNet is using
  the crop or the full canvas.

### Example Recipes

#### Redraw a face with SDXL / NoobAI inpainting

- Detector: face or anime-face segmentation model
- Inpaint masked content: `Latent nothing`
- Denoising strength: high, often `1.0`
- Inpaint only masked: On for local face work
- ControlNet model: NoobAI inpainting model
- ControlNet preprocessor: matching NoobAI inpaint preprocessor if available
- Use ADetailer crop as ControlNet input: On

Use a segmentation detector when you want the mask to follow the face shape
instead of a rectangle.

#### Redraw hands with Anima LLLite inpainting

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

### Custom Models

Put Ultralytics YOLO detector models in:

```text
models/adetailer
```

The model name should end with `.pt`.

### Acknowledgements and License

ADetailer Custom is a derivative work of upstream ADetailer. Thanks to the
authors and contributors of ADetailer and the related open-source projects that
make this fork possible.

This fork is distributed under the AGPL-3.0 license, following the upstream
project. Copyright notices and license terms for upstream-derived portions
remain with their respective rights holders. This fork is responsible only for
the modifications made in this repository.

Original upstream: https://github.com/Bing-su/adetailer
