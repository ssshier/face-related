import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def face_detection(filename):
    image = cv2.imread(filename)
    width = 640
    height = int(image.shape[0] * width / image.shape[1])
    image = cv2.resize(image, (width, height))

    # Detect humans in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8),
                                            scale=1.05)

    if len(rects) > 0:
        return True, filename
    return False, filename


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
