import cv2
import numpy as np


def face_detection(filename):
    # 加载人脸检测器
    model = cv2.dnn.readNetFromTensorflow(
        'data_dnn/opencv_face_detector_uint8.pb',
        'data_dnn/opencv_face_detector.pbtxt')

    # 读取图片
    image = cv2.imread(filename)

    # 构建 blob 并进行前向传播
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0),
                                 swapRB=True, crop=False)
    model.setInput(blob)
    detections = model.forward()

    # 在图片上绘制矩形框标记人脸
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # 设置置信度阈值
            return True, filename
    return False, filename
    #         box = detections[0, 0, i, 3:7] * np.array(
    #             [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
    #         (startX, startY, endX, endY) = box.astype("int")
    #         cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)
    #
    # # 显示结果
    # cv2.imshow('Detected Faces', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    import time
    start = time.time()
    print(face_detection("img/T1.jpeg"))
    print(face_detection("img/T2.jpeg"))
    print(face_detection("img/T3.jpeg"))
    print(face_detection("img/T4.jpeg"))
    print(face_detection("img/T5.jpeg"))
    print(face_detection("img/T6.jpeg"))
    print(face_detection("img/F1.jpeg"))
    print(face_detection("img/F2.jpeg"))
    print(face_detection("img/F3.jpeg"))

    print(time.time() - start)
