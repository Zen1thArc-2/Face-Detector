import cv2

face_frontal_alt2 = (
    "frontalface_alt2.xml"
)
ref = cv2.CascadeClassifier(face_frontal_alt2)
camera = cv2.VideoCapture(0)


def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minSize=(500, 500),minNeighbors=3)
    return faces


def box(frame):
    for x, y, w, h in face_detection(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)


def close_program():
    camera.release()
    cv2.destroyAllWindows()
    exit()


def main():
    while True:
        _, frame = camera.read()
        box(frame)
        cv2.imshow("Facesz", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            close_program()


if __name__ == "__main__":
    main()
