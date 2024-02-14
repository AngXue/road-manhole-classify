import os
import xml.etree.ElementTree as ET
from PIL import Image

# 类别映射，确保这与你的dataset.yaml中的类别对应
class_mapping = {'good': 0, 'broke': 1, 'lose': 2, 'uncovered': 3, 'circle': 4}


def convert_xml_to_yolo(xml_file_path, img_dir, output_dir):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # 提取文件名，用于生成相同名称的.txt文件
    filename = root.find('filename').text
    img_path = os.path.join(img_dir, filename)
    base_filename = os.path.splitext(filename)[0]
    txt_filename = f"{base_filename}.txt"

    # 如果图像尺寸信息在XML中不可用，使用Pillow读取图像尺寸
    img_width, img_height = Image.open(img_path).size

    output_lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_id = class_mapping[class_name]

        xmlbox = obj.find('bndbox')
        xmin = int(xmlbox.find('xmin').text)
        ymin = int(xmlbox.find('ymin').text)
        xmax = int(xmlbox.find('xmax').text)
        ymax = int(xmlbox.find('ymax').text)

        # 转换为YOLO格式
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        output_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")

    # 写入标注文件
    output_path = os.path.join(output_dir, txt_filename)
    with open(output_path, 'w') as file:
        file.write('\n'.join(output_lines))


def batch_convert_xmls(xml_dir, img_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            xml_file_path = os.path.join(xml_dir, xml_file)
            convert_xml_to_yolo(xml_file_path, img_dir, output_dir)
            print(f"Processed {xml_file}")


# 配置你的路径
base_dir = "/home/angxue/Program/Python_Projects/road-manhole-classify/DataSet"
xml_dirs = {
    "train": "train_xmls",
    "val": "val_xmls",
}
img_dirs = {
    "train": "images/train",
    "val": "images/val",
}
label_dirs = {
    "train": "labels/train",
    "val": "labels/val",
}

for key in ["train", "val"]:
    xml_dir = os.path.join(base_dir, xml_dirs[key])
    img_dir = os.path.join(base_dir, img_dirs[key])
    output_dir = os.path.join(base_dir, label_dirs[key])
    batch_convert_xmls(xml_dir, img_dir, output_dir)
