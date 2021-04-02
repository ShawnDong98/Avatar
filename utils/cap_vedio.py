import cv2
import time

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter("mp.avi", fourcc, 30, (640, 480))

start_time = time.time()
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("test", frame)
        writer.write(frame)
        key = cv2.waitKey(1) & 0xff
        time_consume = time.time() - start_time
        print(time_consume)
        if (key == ord('q')) or time_consume > 120:
            print("Quit Process ")
            break
    else:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()