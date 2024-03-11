import cv2
import numpy as np
import onnxruntime as ort


# 可视化ONNX模型
# netron.start("best-sim.onnx")


def load_image(img_path, size=640):
    # 使用相同的预处理方法，YOLOv5在训练时使用的
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))
    image = image.astype(np.float32) / 255.0  # 归一化到[0,1]
    image = np.transpose(image, (2, 0, 1))  # 调整通道顺序为CHW
    image = np.expand_dims(image, axis=0)  # 添加批量维度BCHW
    return image


# 推理函数
def run_inference(onnx_model_path, image_path):
    ort_session = ort.InferenceSession(onnx_model_path)

    # 获取输入和输出张量的名称
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # 加载和预处理图像
    image = load_image(image_path)

    # 进行推理
    outputs = ort_session.run([output_name], {input_name: image})

    # 处理输出
    return outputs[0]


def postprocess_with_objectness(output, conf_threshold=0.25, nms_threshold=0.45):
    output = output[0]  # 假设batch size为1

    boxes = []
    confidences = []
    class_ids = []

    for detection in output:
        objectness_score = detection[4]  # 对象存在概率
        scores = detection[5:]  # 类别得分
        class_id = np.argmax(scores)
        confidence = scores[class_id] * objectness_score

        if confidence > conf_threshold:
            center_x, center_y, width, height = detection[0:4]
            x_min = int(center_x - (width / 2))
            y_min = int(center_y - (height / 2))
            x_max = int(center_x + (width / 2))
            y_max = int(center_y + (height / 2))

            boxes.append([x_min, y_min, x_max, y_max])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # 根据返回的indices类型来适当处理
    if len(indices) == 0:
        final_indices = []  # 没有边界框被选中
    else:
        if isinstance(indices, np.ndarray):
            final_indices = indices.flatten().tolist()  # 转换为Python列表
        elif isinstance(indices[0], tuple):
            final_indices = [i[0] for i in indices]  # 解包元组
        else:
            final_indices = indices  # 直接使用indices

    final_boxes = [boxes[i] for i in final_indices]
    final_class_ids = [class_ids[i] for i in final_indices]
    final_confidences = [confidences[i] for i in final_indices]

    return final_boxes, final_class_ids, final_confidences


def draw_boxes(path):
    # 加载原始图像
    image = cv2.imread(path)

    # 假设 final_class_ids 和 final_confidences 分别包含了类别ID和置信度
    class_names = ['good', 'broke', 'lose', 'uncovered', 'circle']  # 类别名称

    # 假设原始图像尺寸
    orig_height, orig_width = image.shape[:2]  # 使用cv2.imread加载的原始图像尺寸
    model_height, model_width = 640, 640  # 模型输入尺寸

    # 计算缩放因子
    scale_x = orig_width / model_width
    scale_y = orig_height / model_height

    # 对final_boxes中的每个边界框坐标进行反缩放
    scaled_final_boxes = []
    for box in final_boxes:
        x_min, y_min, x_max, y_max = box
        x_min = int(x_min * scale_x)
        y_min = int(y_min * scale_y)
        x_max = int(x_max * scale_x)
        y_max = int(y_max * scale_y)
        scaled_final_boxes.append([x_min, y_min, x_max, y_max])

    # 使用反缩放后的边界框进行绘制
    for box, class_id, confidence in zip(scaled_final_boxes, final_class_ids, final_confidences):
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"{class_names[class_id]}: {confidence:.2f}"
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite("scaled_detection_result.jpg", image)


# 以ONNX模型路径和图像路径为参数运行推理
if __name__ == '__main__':
    path = "/home/angxue/Program/Python_Projects/road-manhole-classify/NewYolovDataSet/images/train/well1_0017.jpg"
    pred = run_inference("best-sim.onnx", path)
    final_boxes, final_class_ids, final_confidences = postprocess_with_objectness(pred)
    draw_boxes(path)
