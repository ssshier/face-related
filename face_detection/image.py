import cv2
import numpy as np

# 假设您已经将图像文件读取到了image_bytes变量中，可以使用open函数从文件路径读取，或者通过其他方式获取图像数据

# 读取图像文件并转换为numpy数组
# 假设image_bytes是二进制图像数据，可以通过open函数从文件路径读取，或者通过其他方式获取图像数据
with open("img/", 'rb') as file:
    image_bytes = file.read()

# 通过numpy数组创建图像
nparr = np.frombuffer(image_bytes, np.uint8)
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
