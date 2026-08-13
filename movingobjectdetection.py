import cv2
cap = cv2.VideoCapture(0)
ret, previous_frame = cap.read()
if not ret:
    print("Cannot access webcam")
    exit()
previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
previous_frame = cv2.GaussianBlur(previous_frame, (21, 21), 0)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    difference = cv2.absdiff(previous_frame, gray)
    _, threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
    threshold = cv2.dilate(threshold, None, iterations=2)
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    movement_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue

        movement_detected = True
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
    if movement_detected:
        cv2.putText(
            frame,
            "Movement Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
    else:
        cv2.putText(
            frame,
            "No Movement",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
    cv2.imshow("Moving Object Detection", frame)
    previous_frame = gray
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
