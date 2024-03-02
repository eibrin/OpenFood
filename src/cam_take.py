import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening video stream or file")
    exit(-1)
else:
    while True:
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow("camera", frame)
            if cv2.waitKey(1) != -1:
                cv2.imwrite("frame.png", frame)
                break
        else:
            print("Error")
            break

cap.release()
cv2.destroyAllWindows()