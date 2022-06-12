from person_tracker import PersonTracker
import cv2

PT = PersonTracker()

cap = cv2.VideoCapture(0)

while cap.isOpened():
	
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue
    
    _, res = PT.process(frame)
    cv2.imshow('test', res)

    if cv2.waitKey(5) == ord("q"):
        cap.release()

cv2.destroyAllWindows()