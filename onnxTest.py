import numpy as np
import tkinter as tk
import subprocess
import netron
import onnxruntime as ort

from PIL import Image
from tkinter import filedialog

# 可视化ONNX模型
netron.start("best-sim.onnx")

# 加载ONNX模型
sess = ort.InferenceSession("best-sim.onnx")


# 加载并预处理图片
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((640, 640))  # 假设模型输入为640x640
    image = np.array(image).astype('float32')  # 转换为浮点数组
    image = np.transpose(image, [2, 0, 1])  # 重排数组维度为CHW
    image = np.expand_dims(image, axis=0)  # 添加批处理维度
    return image


def select_file():
    try:
        # 调用zenity命令选择文件
        process = subprocess.run(['zenity', '--file-selection'], check=True, stdout=subprocess.PIPE,
                                 universal_newlines=True)
        # 获取选中的文件路径
        filepath = process.stdout.strip()
        return filepath
    except subprocess.CalledProcessError:
        # 用户取消选择或命令执行出错
        print("没有选择文件。")
        return None


# 预处理图片
# image_path = select_file()
image_path = "/home/angxue/Program/Python_Projects/road-manhole-classify/DataSet/images/train/well0_0009.jpg"
input_data = preprocess_image(image_path)

# 执行推理
inputs = {sess.get_inputs()[0].name: input_data}
outputs = sess.run(None, inputs)

print(outputs[0])
