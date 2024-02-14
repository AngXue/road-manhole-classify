import argparse
import cv2
import os
import sys
import subprocess
import torch
import tkinter as tk

# 导入YOLOv5模块
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (check_img_size, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from pathlib import Path
from tkinter import filedialog

# 确保YOLOv5的根目录正确添加到sys.path
yolov5_root = '/home/angxue/Program/Python_Projects/road-manhole-classify/yolov5'
sys.path.append(yolov5_root)


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


def run(weights='yolov5/runs/train/exp4/weights/best.pt',  # 模型权重文件
        img_size=640,  # 输入图像大小
        conf_thres=0.25,  # 置信度阈值
        view_img=True):  # 是否显示图像

    source = select_file()  # 使用文件选择器来选择图像源文件
    if not source:
        print("没有选择文件。")
        return

    # 初始化
    device = select_device('')
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(img_size, s=stride)  # 检查图像大小

    # 加载图像
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # 推理
    for path, img, im0s, *rest in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0  # 图像归一化
        if len(img.shape) == 3:
            img = img[None]  # 扩展维度

        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres=0.45, classes=None, agnostic=False)

        # 绘制结果
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if view_img:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        Annotator(im0s, line_width=3, example=str(names)).box_label(xyxy, label,
                                                                                    color=colors(int(cls), True))
                if view_img:
                    cv2.imshow('Detection Result', im0s)
                    # 等待用户按键，如果用户按下'q'键，则退出
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
                        exit()  # 终止程序


if __name__ == '__main__':
    run()
