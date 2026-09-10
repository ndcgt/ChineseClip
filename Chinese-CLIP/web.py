import gradio as gr
import os
import lmdb
import base64
import argparse
import subprocess
import json
import faiss
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import onnxruntime
import numpy as np
import torch
import cn_clip.clip as clip
from cn_clip.clip.utils import  _MODEL_INFO, image_transform
import logging
from openai import OpenAI
import mysql.connector
import bcrypt
import lmdb
logging.getLogger('onnxruntime').setLevel(logging.ERROR)
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)
model = ''
import pandas as pd

# 连接到 MySQL 数据库（配置通过环境变量注入，见 .env.example）
mydb = mysql.connector.connect(
    host=os.environ.get("MYSQL_HOST", "localhost"),
    user=os.environ.get("MYSQL_USER", "root"),
    password=os.environ.get("MYSQL_PASSWORD", ""),
    database=os.environ.get("MYSQL_DATABASE", "clip"),
)

mycursor = mydb.cursor()

mydb.commit()

form_radio = {
    'choices': ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50','self-trained'],
    'value': 'ViT-B-16',
    'interactive': True,
    'label': '多模态模型选择'
}

pic_num = {
   'minimum': 0,
   'maximum': 100,
    'value': 10,
   'step': 5,
    'interactive': True,
    'label': '返回图片数(可能被过滤部分)'
}

top_p_slider = {
   'minimum': 0,
   'maximum': 1,
    'value': 0.1,
   'step': 0.05,
    'interactive': True,
    'label': 'Top P (样本多样性)'
}

small_pic = {
    'choices': ['是', '否'],
    'value': '是',
    'interactive': True,
    'label': '是否返回缩略图'
}

julei = {
    'choices': ['faiss+HNSW', 'FNN'],
    'value': 'faiss+HNSW',
    'interactive': True,
    'label': '寻优模型选择'
}


def register(username, password):
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        mycursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed))
        mydb.commit()
        return "注册成功！请登录。"
    except mysql.connector.IntegrityError:
        return "用户名已存在，请选择其他用户名。"
    except mysql.connector.Error as err:
        return f"注册时发生错误: {err}"


def login(username, password):
    mycursor.execute("SELECT password FROM users WHERE username = %s", (username,))
    result = mycursor.fetchone()
    if result:
        stored_password = result[0]
        # 将存储的密码从字符串转换为字节类型
        if isinstance(stored_password, str):
            stored_password = stored_password.encode('utf-8')
        if bcrypt.checkpw(password.encode('utf-8'), stored_password):
            return True
    return False


def delete_user(username):
    try:
        mycursor.execute("DELETE FROM users WHERE username = %s", (username,))
        mydb.commit()
        if mycursor.rowcount > 0:
            return "用户删除成功。"
        else:
            return "未找到该用户，删除失败。"
    except mysql.connector.Error as err:
        return f"删除用户时发生错误: {err}"

def create_component(params, comp='Slider'):
    if comp == 'Slider':
        return gr.Slider(
            minimum=params['minimum'],
            maximum=params['maximum'],
            value=params['value'],
            step=params['step'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Radio':
        return gr.Radio(
            choices=params['choices'],
            value=params['value'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Button':
        return gr.Button(
            value=params['value'],
            interactive=True
        )
    return None


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


def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("命令执行成功，输出如下：")
        print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，错误信息如下：")
        print(e.stderr)
        return None


def read_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
                except json.JSONDecodeError:
                    print(f"解析以下行时出错: {line}")
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    return data


def open_lmdb(lmdb_path):
    try:
        env = lmdb.open(lmdb_path, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        return env
    except lmdb.Error as e:
        print(f"打开LMDB数据库时出错: {e}")
        return None


def get_image_from_lmdb(txn, image_id):
    try:
        image_b64 = txn.get(str(image_id).encode('utf-8')).tobytes()
        img = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))
        return img
    except (AttributeError, base64.binascii.Error, OSError) as e:
        print(f"获取图像 {image_id} 时出错: {e}")
        return None


def text_search_button_clicked(_question, top_k, models, julei, top_p):
    with open('test_texts.jsonl', 'w', encoding='utf-8') as f:
        json.dump({"text_id":1,"text": _question,"image_ids":[]}, f)
        f.write('\n')

    # 设置PYTHONPATH
    new_path = os.path.dirname(os.path.abspath(__file__))
    os.environ['PYTHONPATH'] = new_path

    # 执行第一个命令
    command1 = [
        'python',
        '-u',
        'cn_clip/eval/extract_features.py',
        '--extract-text-feats',
        '--text-data',
        'test_texts.jsonl',
        '--text-batch-size',
        '1',
        '--context-length',
        '52',
        '--resume',
        f'clip_cn_{models}.pt',
        '--vision-model',
        f'{models}',
        '--text-model',
        'RoBERTa-wwm-ext-base-chinese'
    ]
    run_command(command1)
    if julei=='FNN':

    # 执行第二个命令
        command2 = [
            'python',
            '-u',
            'cn_clip/eval/make_topk_predictions.py',
            f'--image-feats=MUGE/datasets/MUGE/train_imgs.img_feat.jsonl',
            '--text-feats=test_texts.txt_feat.jsonl',
            f'--top-k={top_k}',
            '--eval-batch-size=360000',
            '--output=text_test_predictions.jsonl'
        ]
        run_command(command2)

        # 读取JSONL文件
        file_path = 'text_test_predictions.jsonl'
        jsonl_data = read_jsonl(file_path)
        if jsonl_data:
            image_ids = jsonl_data[0]['image_ids']

            # 创建 image 文件夹
            if not os.path.exists('image'):
                os.makedirs('image')

            # 打开LMDB数据库
            lmdb_imgs = "./MUGE/datasets/MUGE/lmdb/train/imgs"
            with open_lmdb(lmdb_imgs) as env_imgs:
                if env_imgs:
                    txn_imgs = env_imgs.begin(buffers=True)
                    file_list = []
                    for image_id in image_ids:
                        img = get_image_from_lmdb(txn_imgs, image_id)
                        if img:
                            img_path = os.path.join('image', f"{image_id}.png")
                            img.save(img_path)
                            file_list.append((img_path, f"{image_id}"))
                    return gr.Gallery(file_list, columns=3)
        return gr.Gallery([], columns=3)
    else:
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
                '--top-p',
                type=float,
                default=0.1,
                help="Specify the k value of top-p predictions."
            )
            parser.add_argument(
                '--output',
                type=str,
                required=True,
                help="Specify the output jsonl prediction filepath."
            )
            return parser.parse_args(['--image-feats=MUGE/datasets/MUGE/train_imgs.img_feat.jsonl',
                                      '--text-feats=test_texts.txt_feat.jsonl',
                                      f'--top-k={top_k}',
                                      f'--top-p={top_p}',
                                      '--output=text_test_predictions.jsonl'])

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
                with open(args.image_feats, "r") as fin:
                    for line in tqdm(fin):
                        obj = json.loads(line.strip())
                        image_ids.append(obj['image_id'])
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
                print(f"Error: The file {args.image_feats} was not found.")
                exit(1)
            except json.JSONDecodeError:
                print(f"Error: Failed to decode JSON data in {args.image_feats}.")
                exit(1)

            print("Begin to compute top-{} predictions for texts...".format(args.top_k))
            try:
                with open(args.output, "w") as fout:
                    print(args.output)
                    with open(args.text_feats, "r") as fin:
                        for line in tqdm(fin):
                            try:
                                obj = json.loads(line.strip())
                                text_id = obj['text_id']
                                text_feat = np.array([obj['feature']], dtype=np.float32)
                                text_feat = np.ascontiguousarray(text_feat)
                                print(f"Text feature shape: {text_feat.shape}")

                                # 检查数据中是否存在无效值
                                if np.isnan(text_feat).any() or np.isinf(text_feat).any():
                                    print("Warning: Text feature contains NaN or inf values.")

                                # 使用 Faiss 的 HNSW 索引进行搜索
                                distances, indices = index.search(text_feat, args.top_k) #top_p
                                top_k_predictions = [image_ids[i] for i in indices[0]]

                                # 保存结果
                                fout.write(
                                    "{}\n".format(json.dumps({"text_id": text_id, "image_ids": top_k_predictions})))
                            except json.JSONDecodeError:
                                print(f"Error: Failed to decode JSON data in {args.text_feats} for line: {line}")
            except FileNotFoundError:
                print(f"Error: The file {args.text_feats} was not found.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

            print("Top-{} predictions are saved in {}".format(args.top_k, args.output))
            print("Done!")
        file_path = 'text_test_predictions.jsonl'
        jsonl_data = read_jsonl(file_path)
        if jsonl_data:
            image_ids = jsonl_data[0]['image_ids']

            # 创建 image 文件夹
            if not os.path.exists('image'):
                os.makedirs('image')

            # 打开LMDB数据库
            lmdb_imgs = "./MUGE/datasets/MUGE/lmdb/train/imgs"
            with open_lmdb(lmdb_imgs) as env_imgs:
                if env_imgs:
                    txn_imgs = env_imgs.begin(buffers=True)
                    file_list = []
                    for image_id in image_ids:
                        img = get_image_from_lmdb(txn_imgs, image_id)
                        if img:
                            img_path = os.path.join('image', f"{image_id}.png")
                            img.save(img_path)
                            file_list.append((img_path, f"{image_id}"))
                    return gr.Gallery(file_list, columns=3)
        return gr.Gallery([], columns=3)


def pic_search_button_clicked(_question, top_k, models, julei, top_p):
    # 将上传的图片保存到本地并添加到 LMDB
    image_path = _question
    image_id = '1'
    lmdb_dir = './'
    single_image_to_lmdb(image_path, image_id, lmdb_dir)

    # 设置PYTHONPATH
    new_path = os.path.dirname(os.path.abspath(__file__))
    os.environ['PYTHONPATH'] = new_path

    # 执行第一个命令
    command1 = [
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
        f'clip_cn_{models}.pt',
        '--vision-model',
        f'{models}',
        '--text-model',
        'RoBERTa-wwm-ext-base-chinese'
    ]
    run_command(command1)

    if julei=='FNN':

    # 执行第二个命令
        command2 = [
            'python',
            '-u',
            'cn_clip/eval/make_topk_predictions_tr.py',
            '--image-feats=test_imgs.img_feat.jsonl',
            '--text-feats=MUGE/datasets/MUGE/train_texts.txt_feat.jsonl',
            f'--top-k={top_k}',
            '--eval-batch-size=32768',
            '--output=image_test_predictions.jsonl'
        ]
        run_command(command2)
    else:
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
                '--top-p',
                type=float,
                default=0.1,
                help="Specify the k value of top-p predictions."
            )
            parser.add_argument(
                '--output',
                type=str,
                required=True,
                help="Specify the output jsonl prediction filepath."
            )
            return parser.parse_args(['--image-feats=test_imgs.img_feat.jsonl',
                                      '--text-feats=MUGE/datasets/MUGE/train_texts.txt_feat.jsonl',
                                      f'--top-k={top_k}',
                                      f'--top-p={top_p}',
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
                                fout.write(
                                    "{}\n".format(json.dumps({"image_id": text_id, "text_ids": top_k_predictions})))
                            except json.JSONDecodeError:
                                print(f"Error: Failed to decode JSON data in {args.image_feats} for line: {line}")
            except FileNotFoundError:
                print(f"Error: The file {args.image_feats} was not found.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

            print("Top-{} predictions are saved in {}".format(args.top_k, args.output))
            print("Done!")

    # 读取JSONL文件
    file_path = 'image_test_predictions.jsonl'
    jsonl_data = read_jsonl(file_path)
    text_id = jsonl_data[0].get('text_ids', [])
    image_file_path = 'MUGE/datasets/MUGE/train_texts.jsonl'
    image_data = read_jsonl(image_file_path)

    # 查找与 text_id 相关的信息
    matching_items = []
    for item in image_data:
        item_text_id = item.get('text_id', [])
        if item_text_id in text_id:
            matching_items.append(item.get('image_ids', []))

    # 创建 image 文件夹
    if not os.path.exists('image'):
        os.makedirs('image')

    # 打开LMDB数据库
    lmdb_imgs = "./MUGE/datasets/MUGE/lmdb/train/imgs"
    with open_lmdb(lmdb_imgs) as env_imgs:
        if env_imgs:
            txn_imgs = env_imgs.begin(buffers=True)
            file_list = []
            for image_id_list in matching_items:
                for image_id in image_id_list:
                    img = get_image_from_lmdb(txn_imgs, image_id)
                    if img:
                        img_path = os.path.join('image', f"{image_id}.png")
                        img.save(img_path)
                        file_list.append((img_path, f"{image_id}"))
            return gr.Gallery(file_list, columns=3)
    return gr.Gallery([], columns=3)


def no_sample_classification(_question, pic_nums, models, top_p):
    img_sess_options = onnxruntime.SessionOptions()
    img_run_options = onnxruntime.RunOptions()
    img_run_options.log_severity_level = 2
    img_onnx_model_path = "MUGE/deploy/vit-h-14.img.fp16.onnx"
    img_session = onnxruntime.InferenceSession(img_onnx_model_path,
                                               sess_options=img_sess_options,
                                               providers=["CUDAExecutionProvider"])

    model_arch = 'ViT-H-14'
    preprocess = image_transform(_MODEL_INFO[model_arch]['input_resolution'])
    image = preprocess(Image.open(_question)).unsqueeze(0)

    label = [
        "猫", "狗", "鸟", "鱼", "牛", "羊", "猪",
        "汽车", "火车", "飞机", "轮船", "自行车", "摩托车",
        "手机", "电脑", "电视", "冰箱", "洗衣机", "空调","家具","家居饰品","床上用品","厨房用品",
        "桌子", "椅子", "沙发", "床", "书架", "衣柜","手机壳","耳机","充电器","移动存储设备",
        "苹果", "香蕉", "橙子", "面包", "牛奶", "鸡蛋", "米饭","零食","饮料","生鲜","粮油","书包",
        "衬衫", "裤子", "裙子", "外套", "鞋子", "帽子","面部护肤","彩妆","身体护理","香水","皮卡丘",
        "房子", "办公楼", "学校", "医院", "超市", "图书馆","书本","手表","人","玩具","内衣","盒子"
    ]
    # 用ONNX模型计算图像侧特征
    image_features = img_session.run(["unnorm_image_features"], {"image": image.cpu().numpy()})[0]
    image_features = torch.tensor(image_features)
    image_features /= image_features.norm(dim=-1, keepdim=True)  # 归一化后的Chinese-CLIP图像特征，用于下游任务

    txt_sess_options = onnxruntime.SessionOptions()
    txt_run_options = onnxruntime.RunOptions()
    txt_run_options.log_severity_level = 2
    txt_onnx_model_path = "MUGE/deploy/vit-h-14.txt.fp16.onnx"
    txt_session = onnxruntime.InferenceSession(txt_onnx_model_path,
                                               sess_options=txt_sess_options,
                                               providers=["CUDAExecutionProvider"])

    # 为输入文本进行分词。序列长度指定为52，需要和转换ONNX模型时保持一致（参见转换时的context-length参数）
    text = clip.tokenize(label, context_length=52)

    # 用ONNX模型依次计算文本侧特征
    text_features = []
    for i in range(len(text)):
        one_text = np.expand_dims(text[i].cpu().numpy(), axis=0)
        text_feature = txt_session.run(["unnorm_text_features"], {"text": one_text})[0]
        text_feature = torch.tensor(text_feature)
        text_features.append(text_feature)
    text_features = torch.squeeze(torch.stack(text_features), dim=1)  # 特征向量stack到一起
    text_features = text_features / text_features.norm(dim=1, keepdim=True)  # 归一化后的Chinese-CLIP文本特征，用于下游任务

    logits_per_image = 100 * image_features @ text_features.t()
    probs = logits_per_image.softmax(dim=-1)

    # 找到概率最高的label的索引
    max_index = torch.argmax(probs).item()
    result_label = label[max_index]

    return f"最匹配的标签是: {result_label}，对应的概率是: {probs[0][max_index].item()}"

def ask_deepseek(question, history):
    try:
        messages = [{"role": "system", "content": "你是一个负责电商产品多模态检索系统的助手"}]
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        answer = response.choices[0].message.content
        new_history = history + [(question, answer)]
        formatted_history = "\n".join([f"用户: {q}\n助手: {a}" for q, a in new_history])
        return formatted_history, new_history
    except Exception as e:
        new_history = history + [(question, f"发生错误: {str(e)}")]
        formatted_history = "\n".join([f"用户: {q}\n助手: {a}" for q, a in new_history])
        return formatted_history, new_history

def read_associated_data_from_txt(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            image_id, base64_data, text,  text_id = line.strip().split('\t')
            data.append((int(image_id), base64_data, text, text_id))
    return data


def save_base64_image(base64_data, filename):
    img_data = base64.b64decode(base64_data)
    with open(filename, 'wb') as f:
        f.write(img_data)
    return filename


def search_by_text_id(text_id):
    txt_file_path = 'associated_data.txt'
    associated_data = read_associated_data_from_txt(txt_file_path)
    for image_id, base64_data, text, text_id1 in associated_data:
        if text_id1 == text_id:
            img_filename = f"temp_{image_id}.jpg"
            img_path = save_base64_image(base64_data, img_filename)
            return text, [img_path]
    return "未找到对应数据", []


def search_by_image_id(image_id):
    txt_file_path = 'associated_data.txt'
    associated_data = read_associated_data_from_txt(txt_file_path)
    for img_id, base64_data, text, text_id in associated_data:
        if img_id == int(image_id):
            img_filename = f"temp_{img_id}.jpg"
            img_path = save_base64_image(base64_data, img_filename)
            return text, [img_path]
    return "未找到对应数据", []


def search_by_text_content(text_content):
    txt_file_path = 'associated_data.txt'
    associated_data = read_associated_data_from_txt(txt_file_path)
    texts = []
    img_paths = []
    for image_id, base64_data, text, text_id in associated_data:
        if text_content in text:
            img_filename = f"temp_{image_id}.jpg"
            img_path = save_base64_image(base64_data, img_filename)
            texts.append(text)
            img_paths.append(img_path)
    if texts:
        combined_text = "\n".join(texts)
        return combined_text, img_paths
    return "未找到对应数据", []


def search_top_n(n):
    txt_file_path = 'associated_data.txt'
    associated_data = read_associated_data_from_txt(txt_file_path)
    texts = []
    img_paths = []
    for i in range(min(int(n), len(associated_data))):
        image_id, base64_data, text, text_id = associated_data[i]
        img_filename = f"temp_{image_id}.jpg"
        img_path = save_base64_image(base64_data, img_filename)
        texts.append(text)
        img_paths.append(img_path)
    combined_text = "\n".join(texts)
    return combined_text, img_paths

with gr.Blocks(css="""
    body {
        background-color: #f4f4f4;
        font-family: Arial, sans-serif;
    }
    /* 题头样式 */
   .header {
        background-color: #007bff;
        color: white;
        text-align: center;
        padding: 20px;
        font-size: 32px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    /* 侧边栏样式 */
   .sidebar {
        background-color: white;
        width: 250px;
        padding: 20px;
        box-shadow: 2px 0 4px rgba(0, 0, 0, 0.1);
    }
    /* 侧边栏标题样式 */
   .sidebar h2 {
        color: #007bff;
        margin-bottom: 10px;
    }
    /* 侧边栏段落样式 */
   .sidebar p {
        color: #333;
        margin-bottom: 20px;
    }
    /* 主内容区域样式 */
   .main-content {
        padding: 20px;
        flex-grow: 1;
    }
    /* 标题样式 */
   .gr-text-h1 {
        color: #007bff;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    /* Tab 容器样式 */
   .gr-tabs {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    /* 每个 Tab 的样式 */
   .gr-tab {
        background-color: #e6f2ff;
        border-radius: 5px;
        margin-bottom: 10px;
        padding: 10px;
    }
    /* 列容器样式 */
   .gr-column {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 10px;
    }
    .small-column {
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 10px;
    min-width: 100px;
    width: 100px;
    max-width: 100px;
    }
    /* 按钮样式 */
   .gr-button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
   .gr-button:hover {
        background-color: #0056b3;
    }
    .custom-small-button {
    color: black;
    border: none;
    border-radius: 5px;
    padding: 2px 4px;
    cursor: pointer;
    transition: background-color 0.3s ease;
    font-size: 0.7em; 
    display: inline-block;
    min-width: 100px;
    width: 100px;
    max-width: 100px;
    }
    .custom-small-button:hover {
        background-color: #0056b3;
    }
    /* 文本框样式 */
   .gr-textbox {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
    }
    /* 滑动条样式 */
   .gr-slider {
        margin-bottom: 15px;
    }
    .history_display {
    height: 300px; /* 设置固定高度 */
    overflow-y: auto; /* 垂直方向溢出时显示滚动条 */
    }
    /* 单选框样式 */
   .gr-radio {
        margin-bottom: 15px;
    }
    /* 图片展示区域样式 */
   .gr-gallery {
        border: none;
    }
""") as demo:
    with gr.Row():
        # 用于存储登录状态
        logged_in = gr.State(False)
        current_username = gr.State("")
        with gr.Column(visible=not logged_in.value) as login_register_section:
            gr.HTML('<div class="header">登陆注册界面</div>')
            with gr.Row():
                with gr.Column(scale=1, min_width=250):
                    with gr.Tab("注册"):
                        username_register = gr.Textbox(label="用户名")
                        password_register = gr.Textbox(label="密码", type="password")
                        register_button = gr.Button("注册")
                        register_output = gr.Textbox(label="注册结果", interactive=False)

                        register_button.click(
                            fn=register,
                            inputs=[username_register, password_register],
                            outputs=register_output
                        )
                with gr.Column(scale=1, min_width=250):
                    with gr.Tab("登录"):
                        username_login = gr.Textbox(label="用户名")
                        password_login = gr.Textbox(label="密码", type="password")
                        login_button = gr.Button("登录")
                        login_output = gr.Textbox(label="登录结果", interactive=False)


        with gr.Column(visible=logged_in.value) as main_section:
            with gr.Row():
                welcome_message = gr.HTML()
                with gr.Column(elem_classes="small-column",min_width=100):
                    logout_button = gr.Button("登出",elem_classes="custom-small-button")
                    delete_button = gr.Button("注销",elem_classes="custom-small-button")

            def on_login(username, password):
                if login(username, password):
                    logged_in.value = True
                    current_username.value = username
                    welcome = f'<div class="header" style="position: relative;">中文CLIP多功能图文搜索与分类系统 <span style="position: absolute; bottom: 0; right: 0; color: white; font-size: 0.5em">欢迎: {username}</span></div>'
                    return welcome, "登录成功！", gr.update(visible=False), gr.update(visible=True)
                else:
                    return "", "用户名或密码错误，请重试。", gr.update(visible=True), gr.update(visible=False)

            def on_logout():
                logged_in.value = False
                current_username.value = ""
                return "", gr.update(visible=True), gr.update(visible=False)

            def on_delete():
                result = delete_user(current_username.value)
                logged_in.value = False
                current_username.value = ""
                return result, "", gr.update(visible=True), gr.update(visible=False)

            login_button.click(
                fn=on_login,
                inputs=[username_login, password_login],
                outputs=[welcome_message, login_output, login_register_section, main_section]
            )

            logout_button.click(
                fn=on_logout,
                inputs=[],
                outputs=[welcome_message, login_register_section, main_section]
            )

            delete_button.click(
                fn=on_delete,
                inputs=[],
                outputs=[login_output, welcome_message, login_register_section, main_section]
            )
            with gr.Row():
                # 侧边栏部分
                with gr.Column(scale=1, min_width=250):
                    gr.HTML('<h2>使用说明</h2>')
                    gr.HTML('<p>在不同的标签页中，按照提示输入相应内容，选择合适的参数，点击搜索按钮即可得到结果。</p>')
                    gr.HTML('<h2>系统介绍</h2>')
                    gr.HTML('<p>本系统基于中文CLIP实现了文到图搜索、图到图搜索以及无样本分类功能。</p>')
                    gr.HTML(
                        '<p>文本搜索:输入文本，计算文本特征，进行KNN检索，计算top-k召回结果，找到数据库最相似图片个体</p>')
                    gr.HTML(
                        '<p>图片搜索:输入图片，将图片转换为Base64字符串，计算图片特征，进行KNN检索，计算top-k召回结果，找到数据库最相似图片个体</p>')
                    gr.HTML('<p>无样本分类:输入图片，直接调用onnx模型计算该图片与各种类型的相似度关系，输出最优类别</p>')
                    gr.HTML('</div>')
                # 主内容区域
                with gr.Column(scale=3, min_width=600) as main_content:
                    with gr.Tab("文到图搜索"):
                        gr.Markdown('<h1 style="text-align: center; font-size: 32px;">文到图搜索</h1>')
                        gr.Markdown(
                            "在下方文本框输入你想要搜索的文本描述，选择相关参数后点击搜索按钮，系统将为你找到匹配的图片。")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=300):
                                txt_message = gr.Textbox(label="请填写文本", lines=3)
                                pic_nums = create_component(pic_num)
                                models = create_component(form_radio, comp='Radio')
                                julei1 = create_component(julei, comp='Radio')
                                top_p = create_component(top_p_slider)
                                search = create_component({'value': '搜索'}, comp='Button')
                            with gr.Column(scale=1):
                                bt_pic = gr.Gallery(label="检索结果为:", columns=5, show_label=True)
                        search.click(
                            text_search_button_clicked,
                            [txt_message, pic_nums, models, julei1, top_p],
                            [bt_pic]
                        )

                    with gr.Tab("图到图搜索"):
                        gr.Markdown('<h1 style="text-align: center; font-size: 32px;">图到图搜索</h1>')
                        gr.Markdown("上传一张图片，选择合适的参数，点击搜索按钮，系统会找出与之相似的图片。")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=50):
                                input_pic = gr.Image(label="图片", sources=['upload'], type="filepath")
                                pic_nums = create_component(pic_num)
                                models = create_component(form_radio, comp='Radio')
                                julei = create_component(julei, comp='Radio')
                                top_p = create_component(top_p_slider)
                                search = create_component({'value': '搜索'}, comp='Button')
                            with gr.Column(scale=1):
                                bt_pic = gr.Gallery(label="检索结果为:", columns=5, show_label=True)
                        search.click(
                            pic_search_button_clicked,
                            [input_pic, pic_nums, models, julei, top_p],
                            [bt_pic]
                        )

                    with gr.Tab("无样本分类"):
                        gr.Markdown('<h1 style="text-align: center; font-size: 32px;">无样本分类</h1>')
                        gr.Markdown("上传一张图片，系统会根据预定义的标签对其进行分类，并显示分类结果及对应概率。")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=50):
                                input_pic = gr.Image(label="图片", sources=['upload'], type="filepath")
                                models = create_component(form_radio, comp='Radio')
                                search = create_component({'value': '搜索'}, comp='Button')
                            with gr.Column(scale=1):
                                result_text = gr.Textbox(label="分类结果", lines=3, show_label=True)
                        search.click(
                            no_sample_classification,
                            [input_pic, pic_nums, models, top_p],
                            [result_text]
                        )
                    with gr.Tab("数据查找"):
                        gr.Markdown('<h1 style="text-align: center; font-size: 32px;">数据查找</h1>')
                        gr.Markdown("查看数据集前n条数据或根据图片文本id、文本内容进行查找。")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=50):
                                text_id_input = gr.Textbox(label="请填写文本id", lines=1)
                                text_search_button = gr.Button("搜索")
                                text_result_text = gr.Textbox(label="检索文本为", lines=3, show_label=True)
                            with gr.Column(scale=1):
                                text_bt_pic = gr.Gallery(label="检索图片为:", columns=3, show_label=True)

                        with gr.Row():
                            with gr.Column(scale=1, min_width=50):
                                text_content_input = gr.Textbox(label="请填写文本内容", lines=1)
                                text_content_search_button = gr.Button("搜索")
                                text_content_result_text = gr.Textbox(label="检索文本为", lines=3, show_label=True)
                            with gr.Column(scale=1):
                                text_content_bt_pic = gr.Gallery(label="检索图片为:", columns=3, show_label=True)

                        with gr.Row():
                            with gr.Column(scale=1, min_width=50):
                                image_id_input = gr.Textbox(label="请填写图片id", lines=1)
                                image_search_button = gr.Button("搜索")
                                image_result_text = gr.Textbox(label="检索文本为", lines=3, show_label=True)
                            with gr.Column(scale=1):
                                image_bt_pic = gr.Gallery(label="检索图片为:", columns=3, show_label=True)

                        with gr.Row():
                            with gr.Column(scale=1, min_width=50):
                                n_input = gr.Textbox(label="请填写需要检索数", lines=1)
                                top_n_search_button = gr.Button("搜索")
                                top_n_result_text = gr.Textbox(label="检索文本为", lines=3, show_label=True)
                            with gr.Column(scale=1):
                                top_n_bt_pic = gr.Gallery(label="检索图片为:", columns=3, show_label=True)

                        text_search_button.click(
                            search_by_text_id,
                            [text_id_input],
                            [text_result_text, text_bt_pic]
                        )

                        text_content_search_button.click(
                            search_by_text_content,
                            [text_content_input],
                            [text_content_result_text, text_content_bt_pic]
                        )

                        image_search_button.click(
                            search_by_image_id,
                            [image_id_input],
                            [image_result_text, image_bt_pic]
                        )

                        top_n_search_button.click(
                            search_top_n,
                            [n_input],
                            [top_n_result_text, top_n_bt_pic]
                        )
                with gr.Column(scale=1, min_width=250):
                    gr.HTML('<h2>DeepSeek电商助手侧边栏</h2>')
                    input_text = gr.Textbox(label="输入框")
                    history_display = gr.Textbox(label="对话回答", interactive=False, elem_id="history_display")
                    history_state = gr.State([])
                    input_text.submit(
                        ask_deepseek,
                        inputs=[input_text, history_state],
                        outputs=[history_display, history_state]
                    )
# launch
demo.launch(share=False, debug=True, show_api=False, server_port=8090, server_name="127.0.0.1")

