import os

# 图像和标注文件夹路径（根据实际情况调整）
data_path = "EndDataSet"

images_dir = os.path.join(data_path, "images")
labels_dir = os.path.join(data_path, "labels")

# 类别名称和对应前缀
category_prefix = {
    0: "well0_",
    1: "well1_",
    2: "well2_",
    3: "well3_",
    4: "well4_"
}

# 初始化计数器，为每个类别单独计数
category_counts = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}

# 遍历标注文件夹中的所有文件
for label_filename in os.listdir(labels_dir):
    label_path = os.path.join(labels_dir, label_filename)

    # 确保是.txt文件
    if label_path.endswith(".txt"):
        with open(label_path, 'r') as file:
            first_line = file.readline()
            if first_line:  # 确保标注文件非空
                category_id = int(first_line.split()[0])

                # 如果该类别在我们的列表中
                if category_id in category_prefix:
                    # 生成新的文件名和路径
                    new_filename = f"{category_prefix[category_id]}{category_counts[category_id]:04d}"
                    new_image_path = os.path.join(images_dir, new_filename + '.jpg')
                    new_label_path = os.path.join(labels_dir, new_filename + '.txt')

                    # 构建原始图像和标注文件的路径
                    original_image_path = os.path.join(images_dir, label_filename.replace('.txt', '.jpg'))
                    original_label_path = label_path

                    # 重命名图像文件
                    if os.path.exists(original_image_path):
                        os.rename(original_image_path, new_image_path)
                        print(f"Renamed {original_image_path} to {new_image_path}")

                    # 重命名标注文件
                    if os.path.exists(original_label_path):
                        os.rename(original_label_path, new_label_path)
                        print(f"Renamed {original_label_path} to {new_label_path}")

                    # 更新计数器
                    category_counts[category_id] += 1
