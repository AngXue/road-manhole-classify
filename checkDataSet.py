import cv2
import os

from matplotlib import pyplot as plt

# 数据集的根目录路径
dataset_path = 'DataSet'

# 类别名称
class_names = ['good', 'broke', 'lose', 'uncovered', 'circle']

# 训练集图像相对路径
train_images_path = os.path.join(dataset_path, 'images/train')

# 获取训练集图像文件列表
image_files = [f for f in os.listdir(train_images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(train_images_path, image_file)
    label_file = image_file.rsplit('.', 1)[0] + '.txt'
    label_path = os.path.join(dataset_path, 'labels/train', label_file)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        continue
    height, width = image.shape[:2]

    # 读取并绘制标注
    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            for line in file:
                class_id, x_center, y_center, w, h = [float(x) for x in line.split()]
                x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height
                x_min, y_min = int(x_center - w / 2), int(y_center - h / 2)
                x_max, y_max = int(x_center + w / 2), int(y_center + h / 2)

                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image, class_names[int(class_id)], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)

    # OpenCV读取的图像默认颜色通道顺序是BGR，而matplotlib显示的是RGB，需要转换
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用matplotlib显示图像
    plt.imshow(image_rgb)
    plt.title('Image with Annotations')
    plt.show()

    # 请求用户输入以决定是否继续
    user_input = input("Press 'q' to quit, or any other key to continue: ")
    if user_input == 'q':
        continue
