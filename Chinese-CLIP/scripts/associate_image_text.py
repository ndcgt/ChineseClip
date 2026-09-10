import base64
import json
import pandas as pd
from PIL import Image
from io import BytesIO
import gradio as gr


def read_image_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None)
    df.columns = ['image_id', 'base64_data']
    image_dict = df.set_index('image_id')['base64_data'].to_dict()
    return image_dict


def read_text_data(file_path):
    text_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text_info = json.loads(line)
            text_list.append(text_info)
    return text_list


def associate_image_text(image_dict, text_list):
    result = []
    for text_info in text_list:
        image_id = text_info['image_ids']
        if image_id[0] in image_dict:
            base64_data = image_dict[image_id[0]]
            text = text_info['text']
            result.append((image_id[0], base64_data, text, text_info['text_id']))
    return result


def save_associated_data_to_txt(associated_data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for image_id, base64_data, text, text_id in associated_data:
            # 以制表符分隔每个字段，方便后续读取
            line = f"{image_id}\t{base64_data}\t{text}\t{text_id}\n"
            f.write(line)


image_file = 'MUGE/datasets/MUGE/train_imgs.tsv'
text_file = 'MUGE/datasets/MUGE/train_texts.jsonl'

image_dict = read_image_data(image_file)
text_list = read_text_data(text_file)
associated_data = associate_image_text(image_dict, text_list)

# 保存关联结果到 TXT 文件
txt_file_path = 'associated_data.txt'
save_associated_data_to_txt(associated_data, txt_file_path)
print(f"关联结果已保存到 {txt_file_path}")