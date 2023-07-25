import os

import boto3


def face_detection(filename):
    # 配置AWS凭证
    os.environ["AWS_ACCESS_KEY_ID"] = ""
    os.environ["AWS_SECRET_ACCESS_KEY"] = ""
    os.environ["AWS_DEFAULT_REGION"] = ""

    # 创建Rekognition客户端
    rekognition_client = boto3.client('rekognition')

    # 读取图像文件
    with open(filename, 'rb') as image_file:
        image_bytes = image_file.read()

    # 调用人脸检测API
    response = rekognition_client.detect_faces(Image={'Bytes': image_bytes})

    # 解析检测结果
    if 'FaceDetails' in response and response['FaceDetails']:
        face_details = response['FaceDetails']
        print(face_details)
        return True, filename
    return False, filename


if __name__ == '__main__':
    import time
    start = time.time()

    # print(face_detection("img/T1.jpeg"))
    # print(face_detection("img/T2.jpeg"))
    # print(face_detection("img/T3.jpeg"))
    # print(face_detection("img/T4.jpeg"))
    # print(face_detection("img/T5.jpeg"))
    # print(face_detection("img/T6.jpeg"))
    # print(face_detection("img/F1.jpeg"))
    # print(face_detection("img/F2.jpeg"))
    print(face_detection("img/F3.jpeg"))

    print(time.time() - start)
