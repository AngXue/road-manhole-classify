import os
import re
import torch
import shutil
from PIL import Image
from ultralytics import YOLO


def model_run(model_path, data_path, conf=0.35, iou=0.5):
    """
    对给定的模型和数据集运行YOLO对象检测。
    :param iou:
    :param conf:
    :param model_path: 模型路径
    :param data_path: 数据集路径
    :return:
    """
    model = YOLO(model_path)
    return model(data_path, conf=conf, iou=iou, half=True, augment=True, agnostic_nms=True)


def save_results(results, result_txt_path, result_image_path):
    """
    保存所有对象检测结果
    :param results: 检测结果
    :param result_txt_path: 保存结果的txt文件路径
    :param result_image_path: 保存结果的图片文件夹路径
    :return:
    """
    # 通过提取文件名中的数字来排序结果
    results = sorted(results, key=lambda x: int(re.search(r'\d+', x.path.split('/')[-1]).group()))

    with open(result_txt_path, 'a') as file:
        for i, r in enumerate(results):
            im_bgr = r.plot()
            im_rgb = Image.fromarray(im_bgr[..., ::-1])

            # 取完整路径的文件名
            file_name = r.path.split('/')[-1]

            save_path = os.path.join(result_image_path, f'{file_name}')
            im_rgb.save(save_path)

            data = [
                {
                    'cls': int(box.cls.item()),
                    'conf': box.conf.item(),
                    'xyxy': [round(coordinate) for coordinate in box.xyxy.tolist()[0]],
                    'filename': file_name
                }
                for box in r.boxes
            ]

            for item in data:
                # 格式化每行的数据，并以空格分隔
                line = f"{item['filename']}\t{item['cls']}\t{item['conf']}\t{' '.join(map(str, item['xyxy']))}\n"
                # 写入文件
                file.write(line)

            print(f'{i + 1}: {save_path}')
            # r.show()


if __name__ == '__main__':
    # model_path = '井盖测试集/best.pt'
    model_path = '/home/angxue/Downloads/the-end2.pt'
    data_path = '井盖测试集/测试集图片'
    # data_path = '井盖测试集/测试集图片/test10.jpg'

    results = model_run(model_path, data_path)

    result_dir = 'test_results'
    if os.path.exists(result_dir):
        # 删除文件夹及其内容
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    # 保存所有对象检测结果至一个文件
    results_txt = 'results.txt'
    if os.path.exists(results_txt):
        os.remove(results_txt)

    save_results(results, results_txt, result_dir)
