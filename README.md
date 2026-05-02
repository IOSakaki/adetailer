# ADetailer（CN回り改修・プロンプト追記機能実装版）

English summary:

This is a personal experimental fork of Bing-su/adetailer for ReForge / Forge users.  
It adds experimental improvements for using ADetailer with ControlNet, especially Anytest ControlNet models and localized ADetailer inpaint workflows.

This is not the official ADetailer repository.  
Please do not report issues caused by this fork to the upstream ADetailer repository.  
Use this fork at your own risk.

Original upstream: https://github.com/Bing-su/adetailer

---

これは、Bing-su氏による本家ADetailerをもとにした、ReForge / Forge向けの個人用実験フォークです。  
本家ADetailerではありません。

主な目的は、ReForge環境でADetailerとControlNetを組み合わせて使うときの扱いやすさを改善することです。Anytest系ControlNetモデルをADetailer内から直接選びやすくし、顔や手などの局所修正時に、ControlNetがADetailerの局所inpaint処理に合った入力を参照しやすくするための変更を加えています。ADetailerではプロンプト欄を空欄にするとメインのプロンプトをそのまま流用しますが、その末尾に追記できる「Prompt append」欄も追加しています。

## 本家ADetailerとの違い

このフォークでは、主に次の機能を追加しています。

### 1. Anytest系ControlNetモデルへの対応

ADetailerのControlNetモデル選択欄に、`anytest`を含むControlNetモデルが表示されるようにしました。  
これにより、`CN-anytest_v3`などのモデルをADetailer側のControlNet欄から直接選択できます。

### 2. ADetailer cropを使ったControlNet入力

ControlNet欄に`Use ADetailer crop as ControlNet input`のチェックボックスを追加しています。

オンにすると、手動でbbox crop画像をControlNetへ渡すのではなく、ADetailer / ReForge側の既存inpaintパイプラインに委ねます。これにより、顔や手などの局所修正時に、ControlNetが全体キャンバスではなく局所inpaint処理に合った入力を参照しやすくなります。

オフにすると、ControlNetへ元キャンバス画像を明示的に渡し、従来の全体参照に近い挙動にします。

### 3. Prompt append / Negative prompt append

ADetailerのprompt欄を空欄にした場合、従来どおり元画像生成時のpromptを継承します。  
そのうえで、`Prompt append`に書いた内容を最終promptの末尾へ追加できます。

たとえば、ADetailerのprompt欄は空欄のまま、`Prompt append`に次のように書くことで、元のpromptを維持しつつ、顔修正時だけ追加LoRAを適用できます。

```text
<lora:your_face_lora:0.5>
```

同様に、表情だけを変えたい場合は次のように指定できます。

```text
smile, open mouth
```

## Forge / ReForgeへのインストール方法

このフォークは、本家ADetailerと同時に入れず、置き換え用として使ってください。

1. Forge / ReForgeを完全に終了します。
2. WebUIの`extensions`フォルダを開きます。
3. 既に本家ADetailerが入っている場合は、`extensions`フォルダの外へ移動します。
※`adetailer_bak`のように名前を変えただけでは、拡張機能として読み込まれる場合があります。  
バックアップしたい場合は、必ず`extensions`フォルダの外へ移動してください。
4. Forge / ReForgeのExtensions画面から、次のURLを指定してインストールします。

```text
https://github.com/IOSakaki/adetailer
```

手動で入れる場合は、Forge / ReForgeの`extensions`フォルダ内で次を実行します。

```bash
git clone https://github.com/IOSakaki/adetailer.git
```

5. Forge / ReForgeを完全に再起動します。

## SAM3 Text Detector（実験的）

このフォークでは、SAM3を使った**テキスト指定マスク生成**を実験的に利用できます。  
例: `girl face`, `head of anime girl`, `hair`, `eyes`

- 推奨配置先: `models/sam3/`
- SAM3重みは、Meta公式またはHugging Faceから**利用者自身で取得**してください。
- このフォークはSAM3重みを**同梱・再配布しません**。
- SAM3が未導入でも、通常のADetailer（YOLO / MediaPipe / ControlNet）はそのまま利用できます。
- SAM3重みの**自動ダウンロードは行いません**。

### SAM3重みの手動導入手順

1. 使いたいSAM3重みをMeta公式またはHugging Faceからダウンロードします。
2. WebUIルート配下の `models/sam3/` ディレクトリを作成します（存在しない場合）。
3. ダウンロードした重みファイル（例: `sam3_*.pt` など）を `models/sam3/` に配置します。
4. ADetailer detectorで `sam3_text` を選択し、`ADetailer detector classes` にテキストプロンプトを入力します。
5. 必要に応じて `SAM3 target selection` / `SAM3 minimum mask area` を設定します。

> 注: SAM3が見つからない場合でも、ADetailer全体の起動を止めない設計を維持します。

## 注意

これは実験版フォークであり、本家ADetailerではありません。
このフォークで発生した不具合を、本家ADetailer側へ報告しないでください。  
不具合が出た場合は、まずこのフォークを外し、本家ADetailerで再現するか確認してください。
ライセンスは本家ADetailerに従い、AGPL-3.0です。

---


# ADetailer

ADetailer is an extension for the stable diffusion webui that does automatic masking and inpainting. It is similar to the Detection Detailer.

## Install

You can install it directly from the Extensions tab.

![image](https://i.imgur.com/qaXtoI6.png)

Or

(from Mikubill/sd-webui-controlnet)

1. Open "Extensions" tab.
2. Open "Install from URL" tab in the tab.
3. Enter `https://github.com/Bing-su/adetailer.git` to "URL for extension's git repository".
4. Press "Install" button.
5. Wait 5 seconds, and you will see the message "Installed into stable-diffusion-webui\extensions\adetailer. Use Installed tab to restart".
6. Go to "Installed" tab, click "Check for updates", and then click "Apply and restart UI". (The next time you can also use this method to update extensions.)
7. Completely restart A1111 webui including your terminal. (If you do not know what is a "terminal", you can reboot your computer: turn your computer off and turn it on again.)

## Options

| Model, Prompts                    |                                                                                    |                                                                                                                                                        |
| --------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ADetailer model                   | Determine what to detect.                                                          | `None` = disable                                                                                                                                       |
| ADetailer model classes           | Comma separated class names to detect. only available when using YOLO World models | If blank, use default values.<br/>default = [COCO 80 classes](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) |
| ADetailer prompt, negative prompt | Prompts and negative prompts to apply                                              | If left blank, it will use the same as the input.                                                                                                      |
| Skip img2img                      | Skip img2img. In practice, this works by changing the step count of img2img to 1.  | img2img only                                                                                                                                           |

| Detection                            |                                                                                              |              |
| ------------------------------------ | -------------------------------------------------------------------------------------------- | ------------ |
| Detection model confidence threshold | Only objects with a detection model confidence above this threshold are used for inpainting. |              |
| Mask min/max ratio                   | Only use masks whose area is between those ratios for the area of the entire image.          |              |
| Mask only the top k largest          | Only use the k objects with the largest area of the bbox.                                    | 0 to disable |

If you want to exclude objects in the background, try setting the min ratio to around `0.01`.

| Mask Preprocessing              |                                                                                                                                     |                                                                                         |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Mask x, y offset                | Moves the mask horizontally and vertically by                                                                                       |                                                                                         |
| Mask erosion (-) / dilation (+) | Enlarge or reduce the detected mask.                                                                                                | [opencv example](https://docs.opencv.org/4.7.0/db/df6/tutorial_erosion_dilatation.html) |
| Mask merge mode                 | `None`: Inpaint each mask<br/>`Merge`: Merge all masks and inpaint<br/>`Merge and Invert`: Merge all masks and Invert, then inpaint |                                                                                         |

Applied in this order: x, y offset → erosion/dilation → merge/invert.

#### Inpainting

Each option corresponds to a corresponding option on the inpaint tab. Therefore, please refer to the inpaint tab for usage details on how to use each option.

## ControlNet Inpainting

You can use the ControlNet extension if you have ControlNet installed and ControlNet models.

Support `inpaint, scribble, lineart, openpose, tile, depth` controlnet models. Once you choose a model, the preprocessor is set automatically. It works separately from the model set by the Controlnet extension.

If you select `Passthrough`, the controlnet settings you set outside of ADetailer will be used.

## Advanced Options

API request example: [wiki/REST-API](https://github.com/Bing-su/adetailer/wiki/REST-API)

`[SEP], [SKIP], [PROMPT]` tokens: [wiki/Advanced](https://github.com/Bing-su/adetailer/wiki/Advanced)

## Media

- 🎥 [どこよりも詳しい After Detailer (adetailer)の使い方 ① 【Stable Diffusion】](https://youtu.be/sF3POwPUWCE)
- 🎥 [どこよりも詳しい After Detailer (adetailer)の使い方 ② 【Stable Diffusion】](https://youtu.be/urNISRdbIEg)

- 📜 [ADetailer Installation and 5 Usage Methods](https://kindanai.com/en/manual-adetailer/)

## Model

| Model                 | Target                | mAP 50                        | mAP 50-95                     |
| --------------------- | --------------------- | ----------------------------- | ----------------------------- |
| face_yolov8n.pt       | 2D / realistic face   | 0.660                         | 0.366                         |
| face_yolov8s.pt       | 2D / realistic face   | 0.713                         | 0.404                         |
| hand_yolov8n.pt       | 2D / realistic hand   | 0.767                         | 0.505                         |
| person_yolov8n-seg.pt | 2D / realistic person | 0.782 (bbox)<br/>0.761 (mask) | 0.555 (bbox)<br/>0.460 (mask) |
| person_yolov8s-seg.pt | 2D / realistic person | 0.824 (bbox)<br/>0.809 (mask) | 0.605 (bbox)<br/>0.508 (mask) |
| mediapipe_face_full   | realistic face        | -                             | -                             |
| mediapipe_face_short  | realistic face        | -                             | -                             |
| mediapipe_face_mesh   | realistic face        | -                             | -                             |

The YOLO models can be found on huggingface [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer).

For a detailed description of the YOLO8 model, see: https://docs.ultralytics.com/models/yolov8/#overview

YOLO World model: https://docs.ultralytics.com/models/yolo-world/

### Additional Model

Put your [ultralytics](https://github.com/ultralytics/ultralytics) yolo model in `models/adetailer`. The model name should end with `.pt`.

It must be a bbox detection or segment model and use all label.

## How it works

ADetailer works in three simple steps.

1. Create an image.
2. Detect object with a detection model and create a mask image.
3. Inpaint using the image from 1 and the mask from 2.

## Development

ADetailer is developed and tested using the stable-diffusion 1.5 model, for the latest version of [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) repository only.

## License

ADetailer is a derivative work that uses two AGPL-licensed works (stable-diffusion-webui, ultralytics) and is therefore distributed under the AGPL license.

## See Also

- https://github.com/ototadana/sd-face-editor
- https://github.com/continue-revolution/sd-webui-segment-anything
- https://github.com/portu-sim/sd-webui-bmab
