import albumentations as A
import cv2
from pathlib import Path
import shutil
from createDataSet import initialize_dataset_structure, clear_directory, print_dataset_summary


def augment_image(image, augmentation):
    """
    使用Albumentations库对图像进行增强。
    """
    augmented = augmentation(image=image)
    return augmented['image']  # 返回增强后的图像（NumPy数组）


def create_augmented_dataset(original_dir: Path, augmented_dir: Path, augmentation_list):
    """
    对原始数据集进行增强并创建副本数据集，包含原始数据和使用不同增强策略的增强数据。
    """
    clear_directory(augmented_dir)
    initialize_dataset_structure(augmented_dir)

    for image_path in original_dir.glob('images/train/*.*'):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

        # 复制原始图像及其标注到副本数据集
        shutil.copy(image_path, augmented_dir / 'images' / 'train' / image_path.name)
        label_path = original_dir / 'labels' / 'train' / f"{image_path.stem}.txt"
        if label_path.exists():
            shutil.copy(label_path, augmented_dir / 'labels' / 'train' / label_path.name)

        # 为每种增强策略创建增强版本
        for i, augmentation in enumerate(augmentation_list, start=1):
            augmented_image = augment_image(image, augmentation)
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)  # 转换回BGR格式以保存
            augmented_image_name = f"{image_path.stem}_augmented_{i}{image_path.suffix}"
            augmented_image_path = augmented_dir / 'images' / 'train' / augmented_image_name
            cv2.imwrite(str(augmented_image_path), augmented_image)

            # 为增强版本创建对应的标注文件
            augmented_label_name = f"{label_path.stem}_augmented_{i}.txt"
            augmented_label_path = augmented_dir / 'labels' / 'train' / augmented_label_name
            if label_path.exists():
                shutil.copy(label_path, augmented_label_path)


# 定义增强策略列表
augmentation_list = [
    A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    ]),
    A.Compose([
        A.Rotate(limit=30, p=1.0),
    ]),
    A.Compose([
        A.HorizontalFlip(p=1.0),
    ]),
    A.Compose([
        A.VerticalFlip(p=1.0),
    ]),
    A.Compose([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    ]),
    A.Compose([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
    ]),
    A.Compose([
        A.Perspective(scale=(0.05, 0.1), p=1.0),
    ]),
]

if __name__ == '__main__':
    # 示例使用路径
    original_dataset_dir = Path("NewYolovDataSet")
    augmented_dataset_dir = Path("AugmentedDataset")

    # 创建增强后的数据集
    create_augmented_dataset(original_dataset_dir, augmented_dataset_dir, augmentation_list)

    # 打印数据集详细信息
    print("Original Dataset Summary:")
    print_dataset_summary(original_dataset_dir)
    print("\nAugmented Dataset Summary:")
    print_dataset_summary(augmented_dataset_dir)
