# Image & Video Segmentation using SAM 2.1

A tutorial implementation demonstrating promptable, zero-shot object segmentation and multi-frame mask propagation on static images and video sequences using **Segment Anything Model 2.1 (SAM2.1)**, accessed via the Ultralytics inference wrapper.

This is a **learning/practice exercise**, not a production-grade or original research project. The goal is to explore and understand how SAM2.1's promptable segmentation and video mask-propagation APIs work in practice, using pretrained weights and a high-level wrapper library rather than building or training a model from scratch. No original model development, fine-tuning, or novel methodology is involved — the value here is hands-on familiarity with the SAM2 workflow, not a deployable system.


## 1. Overview

This project applies **SAM2.1**, Meta AI's foundation model for promptable visual segmentation, to two distinct tasks:

- **Image segmentation** — generating pixel-accurate object masks from spatial prompts (bounding boxes / points) on a single frame.
- **Video segmentation (object tracking)** — propagating an initial mask/prompt temporally across video frames using SAM2's memory-attention mechanism, without re-prompting on every frame.

Rather than integrating directly against Meta's `sam2` reference repository, the project uses the **Ultralytics `SAM` and `SAM2VideoPredictor` APIs**, which abstract model loading, pre/post-processing, and inference orchestration behind a minimal interface.

## 2. Model Architecture (Background)

SAM2 extends the original Segment Anything Model with the following core components relevant to this implementation:

| Component | Function |
|---|---|
| **Hiera image encoder** | Hierarchical vision transformer backbone that extracts multi-scale image embeddings per frame |
| **Prompt encoder** | Encodes sparse prompts (points, boxes) or dense prompts (masks) into embedding space |
| **Mask decoder** | Produces segmentation masks + IoU/confidence scores from fused image + prompt embeddings |
| **Memory attention module** | Attends to a memory bank of embeddings/masks from previously processed frames to maintain temporal consistency |
| **Memory encoder / bank** | Stores compressed representations of past frame predictions, enabling the model to track an object across frames without re-prompting |

This memory mechanism is what differentiates SAM2 from SAM1: SAM1 is stateless and image-only; SAM2 maintains object identity and mask continuity across a video's temporal dimension.

**Checkpoint used:** `sam2.1_b.pt` (SAM2.1 "base" variant)
- 403 layers
- ~80.85M parameters
- ~80.85M trainable gradients (per `model.info()` output)

## 3. Environment & Dependencies

```bash
pip install -qU ultralytics
```

**Core stack:**
- `ultralytics` (≥8.3.x) — provides `SAM` and `SAM2VideoPredictor` interfaces
- `torch` (CUDA-enabled build recommended)
- `matplotlib` — result visualization

**Reference runtime (as used in the notebook):**
- Ultralytics `8.3.91`
- Python `3.12.3`
- PyTorch `2.5.1`
- CUDA `12.x` on an NVIDIA RTX 4070 (12 GB VRAM)

> GPU inference is strongly recommended. Video mode in particular accumulates frame embeddings in memory/VRAM and is significantly slower on CPU.

## 4. Pipeline

### 4.1 Model Initialization

```python
from ultralytics import SAM

model = SAM('sam2.1_b.pt')
model.info()  # (layers, params, gradients, FLOPs)
```

### 4.2 Image Segmentation (Box-Prompted)

```python
bboxes = [[55, 400, 230, 900]]   # [x1, y1, x2, y2]
image_path = 'test_image.jpg'

results = model(image_path, bboxes=bboxes)
```

- Input is resized/letterboxed internally to the model's working resolution (**1024×1024**).
- Each bounding box is passed to the prompt encoder as a sparse prompt; the decoder returns a binary/soft mask aligned to the original image resolution.
- Multiple boxes can be passed in a single call to segment multiple objects concurrently (demonstrated with a 2-box multi-object case in the notebook, `1 0, 1 1` in inference logs).
- Reported inference performance: **~320–390ms per image** on the reference GPU (preprocess + inference + postprocess).

### 4.3 Video Segmentation (Temporal Mask Propagation)

```python
from ultralytics.models.sam import SAM2VideoPredictor

overrides = dict(
    conf=0.25,
    task='segment',
    mode='predict',
    imgsz=1024,
    model='sam2.1_b.pt'
)
predictor = SAM2VideoPredictor(overrides=overrides)

results = predictor(source='test_video.mp4')
```

- The predictor processes the video **frame-by-frame** (58 frames in the reference run), applying the memory-attention mechanism to propagate object masks forward without requiring a new prompt at every frame.
- `conf=0.25` sets the minimum confidence threshold for retaining a predicted mask/detection.
- **Important operational note:** by default, results accumulate in RAM across all frames. For long videos or large sources, `stream=True` should be passed to return a generator of `Results` objects instead, avoiding out-of-memory failures:

```python
results = predictor(source='test_video.mp4', stream=True)
for r in results:
    boxes = r.boxes   # bounding boxes per frame
    masks = r.masks   # segmentation masks per frame
    probs = r.probs   # class probabilities (if applicable)
```

## 5. Output Objects

Each `Results` object returned by the model exposes:

| Attribute | Description |
|---|---|
| `r.masks` | Per-object segmentation masks (polygon + raster formats) |
| `r.boxes` | Bounding boxes associated with each detected/prompted object |
| `r.probs` | Class probability scores (classification tasks; typically unused in pure segmentation) |

Masks and boxes can be overlaid on the original frames using `matplotlib` for qualitative evaluation, or exported programmatically for downstream tasks (tracking pipelines, dataset annotation, ROI extraction, etc.).

## 6. Performance Notes

| Task | Resolution | Approx. Latency (RTX 4070) |
|---|---|---|
| Single-box image segmentation | 1024×1024 | ~340 ms (34 ms preprocess / 340 ms inference / 13 ms postprocess) |
| Multi-box image segmentation | 1024×1024 | ~320–390 ms |
| Video segmentation | 1024×1024 per frame | ~178 ms/frame (frame 1), variable thereafter |

Actual throughput will vary based on GPU, batch size, number of prompted objects, and whether `stream=True` is used.

