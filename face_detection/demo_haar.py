import cv2


face_cascade = cv2.CascadeClassifier('data_opencv/haarcascades/haarcascade_frontalface_default.xml')


def face_detection(filename):
    image = cv2.imread(filename)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if len(faces) > 0:
        return True, filename
    return False, filename
    # 显示结果
    # cv2.imshow('Detected Faces', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # if len(faces) > 0:
    #     print("exist face")
    # else:
    #     print("not exist face")


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
