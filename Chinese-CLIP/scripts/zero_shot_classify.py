# 完成必要的import（下文省略）
import onnxruntime
from PIL import Image
import numpy as np
import torch
import argparse
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from cn_clip.clip.utils import _MODELS, _MODEL_INFO, _download, available_models, create_model, image_transform

# 载入ONNX图像侧模型（**请替换${DATAPATH}为实际的路径**）
img_sess_options = onnxruntime.SessionOptions()
img_run_options = onnxruntime.RunOptions()
img_run_options.log_severity_level = 2
img_onnx_model_path="MUGE/deploy/vit-h-14.img.fp16.onnx"
img_session = onnxruntime.InferenceSession(img_onnx_model_path,
                                        sess_options=img_sess_options,
                                        providers=["CUDAExecutionProvider"])

# 预处理图片
model_arch = "ViT-H-14" # 这里我们使用的是ViT-B-16规模，其他规模请对应修改
preprocess = image_transform(_MODEL_INFO[model_arch]['input_resolution'])
# 示例皮卡丘图片，预处理后得到[1, 3, 分辨率, 分辨率]尺寸的Torch Tensor
image = preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0)
label = [
        "猫", "狗", "鸟", "鱼", "牛", "羊", "猪",
        "汽车", "火车", "飞机", "轮船", "自行车", "摩托车",
        "手机", "电脑", "电视", "冰箱", "洗衣机", "空调",
        "桌子", "椅子", "沙发", "床", "书架", "衣柜",
        "苹果", "香蕉", "橙子", "面包", "牛奶", "鸡蛋", "米饭",
        "衬衫", "裤子", "裙子", "外套", "鞋子", "帽子",
        "房子", "办公楼", "学校", "医院", "超市", "图书馆"
    ]
# 用ONNX模型计算图像侧特征
image_features = img_session.run(["unnorm_image_features"], {"image": image.cpu().numpy()})[0] # 未归一化的图像特征
image_features = torch.tensor(image_features)
image_features /= image_features.norm(dim=-1, keepdim=True) # 归一化后的Chinese-CLIP图像特征，用于下游任务

txt_sess_options = onnxruntime.SessionOptions()
txt_run_options = onnxruntime.RunOptions()
txt_run_options.log_severity_level = 2
txt_onnx_model_path="MUGE/deploy/vit-h-14.txt.fp16.onnx"
txt_session = onnxruntime.InferenceSession(txt_onnx_model_path,
                                        sess_options=txt_sess_options,
                                        providers=["CUDAExecutionProvider"])

# 为4条输入文本进行分词。序列长度指定为52，需要和转换ONNX模型时保持一致（参见转换时的context-length参数）
text = clip.tokenize(label, context_length=52)

# 用ONNX模型依次计算文本侧特征
text_features = []
for i in range(len(text)):
    one_text = np.expand_dims(text[i].cpu().numpy(),axis=0)
    text_feature = txt_session.run(["unnorm_text_features"], {"text":one_text})[0] # 未归一化的文本特征
    text_feature = torch.tensor(text_feature)
    text_features.append(text_feature)
text_features = torch.squeeze(torch.stack(text_features),dim=1) # 4个特征向量stack到一起
text_features = text_features / text_features.norm(dim=1, keepdim=True) # 归一化后的Chinese-CLIP文本特征，用于下游任务

logits_per_image = 100 * image_features @ text_features.t()
probs = logits_per_image.softmax(dim=-1)
# 找到概率最高的label的索引
max_index = torch.argmax(probs).item()
result_label = label[max_index]
print(r"最匹配的标签是: {result_label}，对应的概率是: {probs[0][max_index].item()}")