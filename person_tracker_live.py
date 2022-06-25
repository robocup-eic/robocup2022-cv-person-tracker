from person_tracker import PersonTracker
import cv2
import time

PT = PersonTracker()

cap = cv2.VideoCapture(2)
start = time.time()

while cap.isOpened():
	
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue
    
    _, res = PT.process(frame)

    fps = 1 / (time.time() - start)
    start = time.time()
    cv2.putText(res, "fps: " + str(round(fps, 2)), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
    cv2.imshow('test', res)

    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
