import cv2
import numpy as np

# 加载 YOLOv3 配置文件和权重文件
net = cv2.dnn.readNet("data_yolo/yolov3.cfg", "data_yolo/yolov3.weights")

# 加载类别标签文件
with open("data_coco/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


def face_detection(filename):
    # 读图片
    image = cv2.imread(filename)

    # 将图片调整到固定宽度（例如640像素），同时保持纵横比
    width = 640
    height = int(image.shape[0] * width / image.shape[1])
    image = cv2.resize(image, (width, height))

    # 获取 YOLOv3 的输出层信息
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # 将图片转换为 blob 格式（用于输入到 YOLOv3模型）
    blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)

    # 将 blob 输入到 YOLOv3 模型中并获取输出
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 处理模型的输出结果
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # 检测框坐标是相对于原始图片大小的比例，所以需要还原到调整后的图片大小
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # 使用 NMS（非最大抑制）来去除重叠的检测框
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # 在图片上绘制检测到的人体框
    for i in indices:
        # i = i[0]
        label = str(classes[class_ids[i]])

        box = boxes[i]
        x, y, w, h = box
        confidence = confidences[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # 显示结果
    cv2.imshow("人体检测", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print(face_detection("img/T6.jpeg"))
