import albumentations as A
import cv2
from pathlib import Path
import shutil
from createDataSet import initialize_dataset_structure, clear_directory, print_dataset_summary


def read_label_file(label_path):
    """
    从YOLO格式的标注文件中读取边界框信息，并确保所有坐标都是归一化的。
    注意：已去除img_width和img_height参数，因为在读取YOLO格式标注时不需要它们。
    """
    boxes = []
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            boxes.append([x_center, y_center, width, height, class_id])
    return boxes


def augment_image_and_labels(image, boxes, class_labels, augmentation):
    """
    使用Albumentations库对图像及其边界框进行增强。
    注意：这里假设boxes和class_labels已经是适当的格式。
    """
    transformed = augmentation(image=image, bboxes=boxes, class_labels=class_labels)
    return transformed['image'], transformed['bboxes'], transformed['class_labels']


def write_label_file(augmented_label_path, augmented_boxes, augmented_class_labels):
    """
    将增强后的边界框和类别标签写入新的标注文件。
    """
    with open(augmented_label_path, 'w') as f:
        for box, class_label in zip(augmented_boxes, augmented_class_labels):
            f.write(f"{class_label} {box[0]} {box[1]} {box[2]} {box[3]}\n")


def create_augmented_dataset(original_dir: Path, augmented_dir: Path, augmentation_list):
    """
    对数据集中的Train部分进行增强并创建副本数据集，包含原始数据和使用不同增强策略的增强数据。
    """
    clear_directory(augmented_dir)
    initialize_dataset_structure(augmented_dir)

    for image_path in original_dir.glob('images/train/*.*'):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        label_path = original_dir / 'labels' / 'train' / f"{image_path.stem}.txt"

        boxes = read_label_file(label_path)
        class_labels = [int(box[4]) for box in boxes]  # 提取类别ID
        # 提取边界框，去除类别ID
        boxes = [box[:4] for box in boxes]

        # 复制原始图像及其标注到副本数据集
        result_path = shutil.copy(image_path, augmented_dir / 'images' / 'train' / image_path.name)
        print(f'{image_path} --> {result_path}')
        label_path = original_dir / 'labels' / 'train' / f"{image_path.stem}.txt"
        if label_path.exists():
            result_path = shutil.copy(label_path, augmented_dir / 'labels' / 'train' / label_path.name)
            print(f'{label_path} --> {result_path}')

        # 为每种增强策略创建增强版本
        for i, augmentation in enumerate(augmentation_list, start=1):
            augmented_image, augmented_boxes, augmented_class_labels = (
                augment_image_and_labels(image, boxes, class_labels, augmentation))
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)  # 转换回BGR格式以保存
            augmented_image_name = f"{image_path.stem}_augmented_{i}{image_path.suffix}"
            augmented_image_path = augmented_dir / 'images' / 'train' / augmented_image_name
            cv2.imwrite(str(augmented_image_path), augmented_image)

            # 为增强版本创建对应的标注文件
            augmented_label_name = f"{label_path.stem}_augmented_{i}.txt"
            augmented_label_path = augmented_dir / 'labels' / 'train' / augmented_label_name
            # 写入新的标注文件
            write_label_file(augmented_label_path, augmented_boxes, augmented_class_labels)

    # 将val和test的数据也复制到新的数据集
    for image_path in original_dir.glob('images/val/*.*'):
        result_path = shutil.copy(image_path, augmented_dir / 'images' / 'val' / image_path.name)
        print(f'{image_path} --> {result_path}')
        label_path = original_dir / 'labels' / 'val' / f"{image_path.stem}.txt"
        if label_path.exists():
            result_path = shutil.copy(label_path, augmented_dir / 'labels' / 'val' / label_path.name)
            print(f'{label_path} --> {result_path}')

    for image_path in original_dir.glob('images/test/*.*'):
        result_path = shutil.copy(image_path, augmented_dir / 'images' / 'test' / image_path.name)
        print(f'{image_path} --> {result_path}')
        label_path = original_dir / 'labels' / 'test' / f"{image_path.stem}.txt"
        if label_path.exists():
            result_path = shutil.copy(label_path, augmented_dir / 'labels' / 'test' / label_path.name)
            print(f'{label_path} --> {result_path}')


# 定义增强策略列表
augmentation_list = [
    A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    A.Compose([
        A.Rotate(limit=60, p=1.0),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    A.Compose([
        A.HorizontalFlip(p=1.0),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    A.Compose([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    A.Compose([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
    A.Compose([
        A.Perspective(scale=(0.05, 0.1), p=1.0),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
]

if __name__ == '__main__':
    # 示例使用路径
    original_dataset_dir = Path("ReplicaSet")
    augmented_dataset_dir = Path("DataSet")

    # 创建增强后的数据集
    create_augmented_dataset(original_dataset_dir, augmented_dataset_dir, augmentation_list)

    # 打印数据集详细信息
    print("Replica Dataset Summary:")
    print_dataset_summary(original_dataset_dir)
    print("\nAugmented Dataset Summary:")
    print_dataset_summary(augmented_dataset_dir)
