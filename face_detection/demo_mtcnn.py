import cv2
from mtcnn.mtcnn import MTCNN


def face_detection(filename):
    detector = MTCNN()
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)

    if len(faces) > 0:
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