# Multimodal Image-Text Retrieval System Based on CLIP

> Built on top of [OFA-Sys/Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP), this project implements a **multimodal image-text retrieval system** for Chinese-language content (graduation project).

The system supports bidirectional image and text feature extraction, **text-to-image / image-to-image** retrieval, **zero-shot image classification**, and user registration and login. It integrates multiple storage and retrieval backends (**FAISS HNSW / Milvus / LMDB / MySQL**) as well as **ONNX / TensorRT** inference acceleration.

---

## Features

- 🖼️ **Image-text feature extraction**: normalized image and text features extracted with Chinese-CLIP (ViT-B-16 / ViT-L-14 / ViT-H-14 / RN50 and others).
- 🔍 **Cross-modal retrieval**:
  - Text-to-image search
  - Image-to-image search
  - FAISS HNSW approximate nearest-neighbour search
- 🏷️ **Zero-shot image classification**: image-text similarity computed from an ONNX model, classifying images without any training.
- 👤 **User system**: registration and login based on MySQL + bcrypt.
- 🤖 **LLM integration**: DeepSeek (OpenAI-compatible API) for text interaction.
- 🗄️ **Multiple storage backends**: FAISS, Milvus, LMDB, MySQL.
- ⚡ **Inference acceleration**: ONNX Runtime and TensorRT deployment (see [deployment.md](deployment.md)).

## System Architecture

```
┌─────────────── Gradio Web UI (web.py) ───────────────┐
│  Register/Login  Text→Image  Image→Image  Zero-shot  │
│                  Search      Search       Classify   │
└──────────────────────────┬───────────────────────────┘
                           │
        ┌──────────────────┼───────────────────┐
        ▼                  ▼                   ▼
   Chinese-CLIP      Vector Search        Storage
   (feature          Engine               Backends
    extraction)   FAISS HNSW / Milvus    LMDB / MySQL
        │                                   │
        ▼                                   ▼
   ONNX / TensorRT accelerated        bcrypt user
   inference                          authentication
```

## Project Structure

```
Chinese-CLIP/
├── web.py                    # Gradio main application (6 functional tabs)
├── scripts/                  # Standalone utility scripts
│   ├── associate_image_text.py   # Associate images with text (writes associated_data.txt)
│   ├── image_search.py           # Image-to-image search (FAISS HNSW)
│   ├── text_search.py            # Text-to-image search (FAISS HNSW)
│   ├── zero_shot_classify.py     # Zero-shot image classification (ONNX)
│   └── milvus.py                 # Milvus vector database example
├── cn_clip/                  # Chinese-CLIP core library (OFA-Sys)
├── run_scripts/              # Fine-tuning / evaluation scripts
├── deployment.md             # ONNX / TensorRT deployment guide
├── Results.md                # Fine-tuning evaluation results
├── dataset_transform.py      # Dataset format conversion
├── requirements.txt          # Dependency list
├── setup.py                  # Packaging configuration
├── .env.example              # Environment variable template (secrets)
└── assets/ examples/         # Static assets and example images
```

## Quick Start

### 1. Prerequisites

- Python 3.8+
- (Optional) PyTorch with CUDA, for GPU training and accelerated inference
- (Optional) A local MySQL database, for the user system
- (Optional) Docker + Milvus, for vector database retrieval

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure secrets

Fill in the real values:

```bash
cp .env.example .env
```

The following variables need to be set:

| Variable | Description |
| --- | --- |
| `DEEPSEEK_API_KEY` | API key for DeepSeek (or any OpenAI-compatible endpoint) |
| `DEEPSEEK_BASE_URL` | Endpoint URL, defaults to `https://api.deepseek.com` |
| `MYSQL_HOST` / `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_DATABASE` | MySQL connection settings |

### 3. Launch the web application

```bash
python web.py
```

The Gradio interface starts at `http://127.0.0.1:8090` with 6 functional tabs:

| Tab | Function |
| --- | --- |
| Register / Login | MySQL + bcrypt user authentication |
| Text-to-Image Search | Enter Chinese text, retrieve the most relevant images |
| Image-to-Image Search | Upload an image, retrieve similar images |
| Zero-shot Classification | Classify an image without training |
| Data Lookup | Browse the first N images in the database |

The Chinese-CLIP pretrained weights are downloaded automatically on first run (ViT-B-16 by default).

## Standalone Scripts

```bash
# Text-to-image search (FAISS HNSW)
python scripts/text_search.py

# Image-to-image search
python scripts/image_search.py

# Zero-shot image classification (ONNX)
python scripts/zero_shot_classify.py

# Associate images with text
python scripts/associate_image_text.py

# Milvus vector database example (requires a running Milvus service)
python scripts/milvus.py
```

## Training and Evaluation

This project fine-tunes on the [MUGE](https://tianchi.aliyun.com/muge) Chinese multimodal image-text dataset. The pipeline is as follows:

```bash
# 1. Fine-tune ViT-B-16 (DATAPATH is the data directory)
bash run_scripts/muge_finetune_vit-b-16_rbt-base.sh ${DATAPATH}

# 2. Extract image and text features
python -u cn_clip/eval/extract_features.py \
    --extract-image-feats --extract-text-feats \
    --image-data="MUGE/datasets/MUGE/lmdb/valid/imgs" \
    --text-data="MUGE/datasets/MUGE/valid_texts.jsonl" \
    --img-batch-size=32 --text-batch-size=32 \
    --context-length=52 \
    --resume=clip_cn_vit-b-16.pt \
    --vision-model=ViT-B-16 \
    --text-model=RoBERTa-wwm-ext-base-chinese

# 3. KNN retrieval (text-to-image / image-to-text)
python -u cn_clip/eval/make_topk_predictions.py \
    --image-feats="MUGE/datasets/MUGE/valid_imgs.img_feat.jsonl" \
    --text-feats="MUGE/datasets/MUGE/valid_texts.txt_feat.jsonl" \
    --top-k=10 --eval-batch-size=32768 \
    --output="MUGE/datasets/MUGE/valid_predictions.jsonl"

python -u cn_clip/eval/make_topk_predictions_tr.py \
    --image-feats="MUGE/datasets/MUGE/valid_imgs.img_feat.jsonl" \
    --text-feats="MUGE/datasets/MUGE/valid_texts.txt_feat.jsonl" \
    --top-k=10 --eval-batch-size=32768 \
    --output="MUGE/datasets/MUGE/valid_tr_predictions.jsonl"

# 4. Compute Recall
python cn_clip/eval/evaluation.py \
    MUGE/datasets/MUGE/valid_texts.jsonl \
    MUGE/datasets/MUGE/valid_predictions.jsonl output.json
```


**MUGE Validation Set Evaluation Results (Fine-tuned ViT-B-16)**：

| Task | R@1 | R@5 | R@10 | mean recall |
| --- | --- | --- | --- | --- |
| text→image | 52.10 | 76.82 | 84.23 | 71.05 |
| image→text | 38.69 | 65.44 | 74.90 | 59.68 |

## Deployment Acceleration (ONNX / TensorRT)

PyTorch models can be converted to ONNX / TensorRT to accelerate inference:

```bash
# Convert to ONNX
python cn_clip/deploy/pytorch_to_onnx.py \
    --model-arch ViT-H-14 \
    --pytorch-ckpt-path MUGE/pretrained_weights/clip_cn_vit-h-14.pt \
    --save-onnx-path MUGE/deploy/vit-h-14 \
    --convert-text --convert-vision

# Further convert to TensorRT (fp16)
python cn_clip/deploy/onnx_to_tensorrt.py \
    --model-arch ViT-H-14 \
    --convert-text --text-onnx-path MUGE/deploy/vit-h-14.txt.fp16.onnx \
    --convert-vision --vision-onnx-path MUGE/deploy/vit-h-14.img.fp16.onnx \
    --save-tensorrt-path MUGE/deploy/vit-h-14 --fp16
```

See [deployment.md](deployment.md) for details.

## Acknowledgements and Citation

This project is built on the open-source project [OFA-Sys/Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) (MIT License). Copyright of its models and code belongs to the original authors. Citation:

```bibtex
@article{chinese-clip,
  title={Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese},
  author={Yang, An and Pan, Junshu and Lin, Junyang and Men, Rui and Zhang, Yichang and Zhou, Jingren and Zhou, Chang},
  journal={arXiv preprint arXiv:2211.01335},
  year={2022}
}
```

## License

This project is built on [OFA-Sys/Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) and follows its [MIT License](MIT-LICENSE.txt).
