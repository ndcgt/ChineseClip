# 基于 Chinese-CLIP 的中文图文多模态检索系统

> 本项目以 [OFA-Sys/Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) 为底座，构建了一套面向中文场景的**图文多模态检索系统**（毕业设计）。

系统支持图像与文本的双向特征提取、**文搜图 / 图搜图**检索、**零样本图像分类**、用户注册登录，并集成了 **FAISS HNSW / Milvus / LMDB / MySQL** 等多种存储与检索方案，以及 **ONNX / TensorRT** 推理加速。

---

## 功能特性

- 🖼️ **图文特征提取**：基于 Chinese-CLIP（ViT-B-16 / ViT-L-14 / ViT-H-14 / RN50 等）提取归一化图文特征。
- 🔍 **跨模态检索**：
  - 文到图搜索（text-to-image）
  - 图到图搜索（image-to-image）
  - 基于 FAISS HNSW 的向量近邻检索
- 🏷️ **零样本图像分类**：基于 ONNX 模型的图文相似度，无需训练即可对图片分类。
- 👤 **用户系统**：基于 MySQL + bcrypt 的注册 / 登录。
- 🤖 **LLM 集成**：接入 DeepSeek（OpenAI 兼容接口）进行文本交互。
- 🗄️ **多存储后端**：FAISS、Milvus、LMDB、MySQL。
- ⚡ **推理加速**：支持 ONNX Runtime 与 TensorRT 部署（详见 [deployment.md](deployment.md)）。

## 系统架构

```
┌─────────────── Gradio Web 界面 (web.py) ───────────────┐
│  注册/登录  文到图搜索  图到图搜索  无样本分类  数据查找  │
└──────────────────────────┬────────────────────────────┘
                           │
        ┌──────────────────┼───────────────────┐
        ▼                  ▼                   ▼
   Chinese-CLIP      向量检索引擎         存储后端
   (特征提取)     FAISS HNSW / Milvus    LMDB / MySQL
        │                                   │
        ▼                                   ▼
   ONNX / TensorRT 加速推理            bcrypt 用户认证
```

## 目录结构

```
Chinese-CLIP/
├── web.py                    # Gradio 主应用（6 个功能 Tab）
├── scripts/                  # 独立功能脚本
│   ├── associate_image_text.py   # 关联图片与文本（生成 associated_data.txt）
│   ├── image_search.py           # 以图搜图（FAISS HNSW）
│   ├── text_search.py            # 以文搜图（FAISS HNSW）
│   ├── zero_shot_classify.py     # 零样本图像分类（ONNX）
│   └── milvus.py                 # Milvus 向量数据库示例
├── cn_clip/                  # Chinese-CLIP 核心库（OFA-Sys）
├── run_scripts/              # 微调 / 评测训练脚本
├── deployment.md             # ONNX / TensorRT 部署说明
├── Results.md                # 微调评测结果
├── dataset_transform.py      # 数据集格式转换
├── requirements.txt          # 依赖清单
├── setup.py                  # 打包配置
├── .env.example              # 环境变量示例（密钥配置）
└── assets/ examples/         # 静态资源与示例图片
```

## 快速开始

### 1. 环境准备

- Python 3.8+
- （可选）CUDA 环境的 PyTorch，用于 GPU 训练 / 推理加速
- （可选）本地 MySQL 数据库，用于用户系统
- （可选）Docker + Milvus，用于向量数据库检索

安装依赖：

```bash
pip install -r requirements.txt
```

> GPU 用户可将 `faiss-cpu` 替换为 `faiss-gpu`、`onnxruntime` 替换为 `onnxruntime-gpu`。

### 2. 配置密钥

复制 `.env.example` 为 `.env`，并填入真实值：

```bash
cp .env.example .env
```

需要配置：

| 变量 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek（或 OpenAI 兼容接口）的 API Key |
| `DEEPSEEK_BASE_URL` | 接口地址，默认 `https://api.deepseek.com` |
| `MYSQL_HOST` / `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_DATABASE` | MySQL 连接信息 |

> ⚠️ 请勿将 `.env` 提交到 Git（已在 `.gitignore` 中排除）。

### 3. 启动 Web 应用

```bash
python web.py
```

默认在 `http://127.0.0.1:8090` 启动 Gradio 界面，包含 6 个功能 Tab：

| Tab | 功能 |
| --- | --- |
| 注册 / 登录 | MySQL + bcrypt 用户认证 |
| 文到图搜索 | 输入中文文本，检索最相关图片 |
| 图到图搜索 | 上传图片，检索相似图片 |
| 无样本分类 | 无需训练，对图片做零样本分类 |
| 数据查找 | 直接浏览数据库中的前 N 张图片 |

首次运行会自动下载 Chinese-CLIP 预训练权重（默认 `ViT-B-16`）。

## 独立脚本使用

```bash
# 以文搜图（FAISS HNSW）
python scripts/text_search.py

# 以图搜图
python scripts/image_search.py

# 零样本图像分类（ONNX）
python scripts/zero_shot_classify.py

# 关联图片与文本
python scripts/associate_image_text.py

# Milvus 向量库示例（需先启动 Milvus 服务）
python scripts/milvus.py
```

## 训练与评测

本项目在 [MUGE](https://tianchi.aliyun.com/muge) 中文多模态图文数据集上微调，主要流程如下：

```bash
# 1. 微调 ViT-B-16（DATAPATH 为数据目录）
bash run_scripts/muge_finetune_vit-b-16_rbt-base.sh ${DATAPATH}

# 2. 提取图文特征
python -u cn_clip/eval/extract_features.py \
    --extract-image-feats --extract-text-feats \
    --image-data="MUGE/datasets/MUGE/lmdb/valid/imgs" \
    --text-data="MUGE/datasets/MUGE/valid_texts.jsonl" \
    --img-batch-size=32 --text-batch-size=32 \
    --context-length=52 \
    --resume=clip_cn_vit-b-16.pt \
    --vision-model=ViT-B-16 \
    --text-model=RoBERTa-wwm-ext-base-chinese

# 3. KNN 检索（文搜图 / 图搜文）
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

# 4. 计算 Recall
python cn_clip/eval/evaluation.py \
    MUGE/datasets/MUGE/valid_texts.jsonl \
    MUGE/datasets/MUGE/valid_predictions.jsonl output.json
```

**评测结果**

上面这条链路用于在 MUGE 官方验证集上评测微调后的模型。本项目未做独立的定量评测，因此不提供自测指标。

上游 OFA-Sys/Chinese-CLIP 在 MUGE 官方验证集上公布的 ViT-B-16 结果为零样本 R@1 52.1 / R@5 76.7 / R@10 84.4，微调后 R@1 58.4 / R@5 83.6 / R@10 90.0，原始表格见仓库内的 [`Results.md`](Results.md)。**这些是上游项目的基准数据，不是本项目的实验结果。**

## 部署加速（ONNX / TensorRT）

支持将 PyTorch 模型转换为 ONNX / TensorRT 以加速推理：

```bash
# 转换为 ONNX
python cn_clip/deploy/pytorch_to_onnx.py \
    --model-arch ViT-H-14 \
    --pytorch-ckpt-path MUGE/pretrained_weights/clip_cn_vit-h-14.pt \
    --save-onnx-path MUGE/deploy/vit-h-14 \
    --convert-text --convert-vision

# 进一步转换为 TensorRT (fp16)
python cn_clip/deploy/onnx_to_tensorrt.py \
    --model-arch ViT-H-14 \
    --convert-text --text-onnx-path MUGE/deploy/vit-h-14.txt.fp16.onnx \
    --convert-vision --vision-onnx-path MUGE/deploy/vit-h-14.img.fp16.onnx \
    --save-tensorrt-path MUGE/deploy/vit-h-14 --fp16
```

详细说明见 [deployment.md](deployment.md)。

## 致谢与引用

本项目基于开源项目 [OFA-Sys/Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)（MIT License），其模型与代码版权归原作者所有。相关引用如下：

```bibtex
@article{chinese-clip,
  title={Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese},
  author={Yang, An and Pan, Junshu and Lin, Junyang and Men, Rui and Zhang, Yichang and Zhou, Jingren and Zhou, Chang},
  journal={arXiv preprint arXiv:2211.01335},
  year={2022}
}
```

## 许可证

本项目在 [OFA-Sys/Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) 基础上构建，遵循其 [MIT 许可证](MIT-LICENSE.txt)。
