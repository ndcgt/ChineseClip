import os
import subprocess
import faiss
from tqdm import tqdm
import json
import argparse
import os
import subprocess
import argparse
import numpy as np
import faiss
from tqdm import tqdm
import json
from io import BytesIO
from PIL import Image
import lmdb
import base64

def single_image_to_lmdb(image_path, image_id, lmdb_dir):
    # 确保 LMDB 目录存在
    os.makedirs(lmdb_dir, exist_ok=True)
    lmdb_img = os.path.join(lmdb_dir, "imgs")
    # 打开 LMDB 环境，设置最大映射大小为 1GB
    env_img = lmdb.open(lmdb_img, map_size=1024 ** 3)
    txn_img = env_img.begin(write=True)

    try:
        # 以二进制模式读取图片
        with open(image_path, 'rb') as f:
            image_binary = f.read()

        # 将图片二进制数据转换为 Base64 字符串
        base64_image = base64.b64encode(image_binary).decode('utf-8')

        # 将图片 ID 编码为字节类型作为键
        key = str(image_id).encode('utf-8')
        # 把图片的 Base64 编码字符串作为值，转换为字节类型
        value = base64_image.encode('utf-8')
        txn_img.put(key=key, value=value)
        # 记录图片数量
        txn_img.put(key=b'num_images', value="1".encode('utf-8'))
        # 提交事务
        txn_img.commit()
        print(f"Finished serializing 1 image's Base64 data into {lmdb_img}.")

    except Exception as e:
        print(f"Error processing image: {e}")
    finally:
        # 关闭 LMDB 环境
        env_img.close()


image_path = 'examples/pokemon.jpeg'
image_id = '1'
lmdb_dir = './'
single_image_to_lmdb(image_path, image_id, lmdb_dir)

new_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['PYTHONPATH'] = new_path
command = [
    'python',
    '-u',
    'cn_clip/eval/extract_features.py',
    '--extract-image-feats',
    '--text-data',
    'test_texts.jsonl',
    '--image-data',
    'imgs',
    '--text-batch-size',
    '1',
    '--context-length',
    '52',
    '--resume',
    'clip_cn_vit-b-16.pt',
    '--vision-model',
    'ViT-B-16',
    '--text-model',
    'RoBERTa-wwm-ext-base-chinese'
]
try:
    result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
    print("命令执行成功，输出如下：")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"命令执行失败，错误信息如下：")
    print(e.stderr)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-feats',
        type=str,
        required=False,
        help="Specify the path of image features."
    )
    parser.add_argument(
        '--text-feats',
        type=str,
        required=False,
        help="Specify the path of text features."
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help="Specify the k value of top-k predictions."
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help="Specify the output jsonl prediction filepath."
    )
    return parser.parse_args(['--image-feats=test_imgs.img_feat.jsonl',
                              '--text-feats=MUGE/datasets/MUGE/train_texts.txt_feat.jsonl',
                              '--top-k=10',
                              '--output=image_test_predictions.jsonl'])


if __name__ == "__main__":
    args = parse_args()

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    print("Begin to load image features...")
    image_ids = []
    image_feats = []
    try:
        with open(args.text_feats, "r") as fin:
            for line in tqdm(fin):
                obj = json.loads(line.strip())
                image_ids.append(obj['text_id'])
                image_feats.append(obj['feature'])
        image_feats_array = np.array(image_feats, dtype=np.float32)
        print(f"Image features array shape: {image_feats_array.shape}")

        # 检查数据中是否存在无效值
        if np.isnan(image_feats_array).any() or np.isinf(image_feats_array).any():
            print("Warning: Image features contain NaN or inf values.")

        image_feats_array = np.ascontiguousarray(image_feats_array)

        # 创建 HNSW 索引并迁移到 GPU
        d = image_feats_array.shape[1]  # 特征维度
        index_cpu = faiss.IndexHNSWFlat(d, 32)  # 32 是每个节点的最大连接数
        index_cpu.hnsw.efConstruction = 30
        index_cpu.hnsw.efSearch = 40

        # 初始化 GPU 资源
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)

        print("Before adding data to index, index is valid: ", index.is_trained)
        index.add(image_feats_array)
        print("After adding data to index, index is valid: ", index.is_trained)

        print("Finished loading image features.")
    except FileNotFoundError:
        print(f"Error: The file {args.text_feats} was not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON data in {args.text_feats}.")
        exit(1)

    print("Begin to compute top-{} predictions for texts...".format(args.top_k))
    try:
        with open(args.output, "w") as fout:
            print(args.output)
            with open(args.image_feats, "r") as fin:
                for line in tqdm(fin):
                    try:
                        obj = json.loads(line.strip())
                        text_id = obj['image_id']
                        text_feat = np.array([obj['feature']], dtype=np.float32)
                        text_feat = np.ascontiguousarray(text_feat)
                        print(f"Text feature shape: {text_feat.shape}")

                        # 检查数据中是否存在无效值
                        if np.isnan(text_feat).any() or np.isinf(text_feat).any():
                            print("Warning: Text feature contains NaN or inf values.")

                        # 使用 Faiss 的 HNSW 索引进行搜索
                        distances, indices = index.search(text_feat, args.top_k)
                        top_k_predictions = [image_ids[i] for i in indices[0]]

                        # 保存结果
                        fout.write("{}\n".format(json.dumps({"image_id": text_id, "text_ids": top_k_predictions})))
                    except json.JSONDecodeError:
                        print(f"Error: Failed to decode JSON data in {args.image_feats} for line: {line}")
    except FileNotFoundError:
        print(f"Error: The file {args.image_feats} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("Top-{} predictions are saved in {}".format(args.top_k, args.output))
    print("Done!")


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                # 解析每行的 JSON 数据
                json_obj = json.loads(line.strip())
                data.append(json_obj)
            except json.JSONDecodeError:
                print(f"解析以下行时出错: {line}")
    return data


file_path = 'image_test_predictions.jsonl'
jsonl_data = read_jsonl(file_path)
text_id = jsonl_data[0].get('text_ids', [])  # 使用 get 方法避免键不存在的错误
image_file_path = 'MUGE/datasets/MUGE/train_texts.jsonl'
image_data = read_jsonl(image_file_path)

# 查找与 text_id 相关的信息
matching_items = []
for item in image_data:
    item_text_id = item.get('text_id', [])
    if item_text_id in text_id:
        matching_items.append(item.get('image_ids',[]))

lmdb_imgs = "./MUGE/datasets/MUGE/lmdb/train/imgs"
env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
txn_imgs = env_imgs.begin(buffers=True)
for image_id in matching_items:
    image_b64 = txn_imgs.get("{}".format(image_id[0]).encode('utf-8')).tobytes()
    img = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))
    img.show()

