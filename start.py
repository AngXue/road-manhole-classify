import sys

from yolov5.train import run  # 导入YOLOv5的训练函数

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model



if __name__ == '__main__':
    # 设置训练参数
    train_params = {
        'img_size': 640,  # 图像大小
        'batch_size': 16,  # 批次大小
        'epochs': 300,  # 训练周期
        'data': 'manhole_dataset.yaml',  # 数据集配置文件
        'weights': 'yolov5m.pt',  # 预训练权重
        'cache': True  # 缓存图像以加速训练
    }

    # 启动训练
    run(**train_params)
