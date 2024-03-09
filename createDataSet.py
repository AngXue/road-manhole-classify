from pathlib import Path
import shutil
import random
import re


def initialize_dataset_structure(dataset_dir: Path):
    """
    确保数据集目录结构存在，包括images/train, images/val, labels/train, labels/val.
    """
    subdirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for subdir in subdirs:
        (dataset_dir / subdir).mkdir(parents=True, exist_ok=True)


def clear_directory(directory: Path):
    """
    清理指定目录，如果目录不存在，则先创建它。
    """
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    else:
        for item in directory.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def create_replica_dataset(original_dir: Path, replica_dir: Path):
    """
    创建副本数据集，包括从原始数据集的训练集中随机抽取部分数据作为验证集。
    """
    clear_directory(replica_dir)
    initialize_dataset_structure(replica_dir)

    # 遍历原始数据集中的图像文件并复制
    for image_path in original_dir.glob('images/train/*.*'):
        category = re.match(r'(well[0-5])_', image_path.name)
        if category:
            # 直接复制图像和标注文件到副本的训练集
            shutil.copy(image_path, replica_dir / 'images' / 'train' / image_path.name)
            label_path = original_dir / 'labels' / 'train' / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, replica_dir / 'labels' / 'train' / label_path.name)

    # 对每个类别，从副本训练集中随机抽取10%的图像作为验证集
    for category in range(6):
        category_files = list(replica_dir.glob(f'images/train/well{category}_*.*'))
        sampled_files = random.sample(category_files, k=max(1, len(category_files) // 20))

        for file in sampled_files:
            shutil.move(file, replica_dir / 'images' / 'val' / file.name)
            label_file = replica_dir / 'labels' / 'train' / f"{file.stem}.txt"
            if label_file.exists():
                shutil.move(label_file, replica_dir / 'labels' / 'val' / label_file.name)


def print_dataset_summary(dataset_dir: Path):
    """
    数据按照 '类别: 训练图像数 | 训练标注数 | 验证图像数 | 验证标注数' 的格式横向打印。
    """
    categories = [f'well{i}' for i in range(6)]
    counts = {category: {'train_images': 0, 'train_labels': 0, 'val_images': 0, 'val_labels': 0} for category in
              categories}

    # 收集计数信息
    for category in categories:
        counts[category]['train_images'] = len(list((dataset_dir / 'images/train').glob(f'{category}_*.*')))
        counts[category]['train_labels'] = len(list((dataset_dir / 'labels/train').glob(f'{category}_*.txt')))
        counts[category]['val_images'] = len(list((dataset_dir / 'images/val').glob(f'{category}_*.*')))
        counts[category]['val_labels'] = len(list((dataset_dir / 'labels/val').glob(f'{category}_*.txt')))

    # 打印汇总信息
    print(f"\nDataset Summary for {dataset_dir}:")
    for category in categories:
        cat_counts = counts[category]
        print(f"{category}: Images (Train | Val) = {cat_counts['train_images']} | {cat_counts['val_images']}, "
              f"Labels (Train | Val) = {cat_counts['train_labels']} | {cat_counts['val_labels']}")


if __name__ == '__main__':
    # 示例使用
    original_dataset_dir = Path("AugmentedDataset")
    replica_dataset_dir = Path("DataSet")

    # 创建副本数据集
    create_replica_dataset(original_dataset_dir, replica_dataset_dir)

    # 打印数据集详细信息
    print("Augmented Dataset Summary:")
    print_dataset_summary(original_dataset_dir)
    print("\nReplica Dataset Summary:")
    print_dataset_summary(replica_dataset_dir)
