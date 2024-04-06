import os
import torch
import shutil
from PIL import Image
from ultralytics import YOLO

model = YOLO('/home/angxue/Downloads/maybe-the-end2.pt')
# model = YOLO('井盖测试集/best.pt')

source = '井盖测试集/测试集图片'
# source = '井盖测试集/测试集图片/test10.jpg'

# results = model(source, stream=True)
results = model(source=source, conf=0.25, iou=0.4, half=True, augment=True, agnostic_nms=True)

result_dir = 'test_results'
if os.path.exists(result_dir):
    # 删除文件夹及其内容
    shutil.rmtree(result_dir)
os.mkdir(result_dir)

# 保存所有对象检测结果至一个文件
results_txt = 'results.txt'
if os.path.exists(results_txt):
    os.remove(results_txt)

# 保存
with open('results.txt', 'a') as file:
    for i, r in enumerate(results):
        im_bgr = r.plot()
        im_rgb = Image.fromarray(im_bgr[..., ::-1])

        # 取完整路径的文件名
        file_name = r.path.split('/')[-1]

        save_path = os.path.join(result_dir, f'{file_name}')
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

        print(f'{i+1}: {save_path}')
        # r.show()
