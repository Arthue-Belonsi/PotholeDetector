import cv2
import numpy as np

net = cv2.dnn.readNetFromDarknet('models/yolov4_tiny_pothole.cfg', 'models/yolov4_tiny_pothole_last.weights')
classes = ['pothole']

#cap = cv2.VideoCapture('http://192.168.0.15:4747/video')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (1200, 800))

    ht, wt, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)

    last_layer = net.getUnconnectedOutLayersNames()
    layer_out = net.forward(last_layer)

    boxes = []
    confidences = []
    cls_ids = []

    for output in layer_out:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.25 and class_id == 0:
                center_x = int(detection[0] * wt)
                center_y = int(detection[1] * ht)

                w = int(detection[2] * wt)
                h = int(detection[3] * ht)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                cls_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.2)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[cls_ids[i]])

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, label, (x, y - 10), font, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"{w}x{h}", (x, y - 30), font, 0.5, (0, 0, 255), 2)

    cv2.imshow('Pothole Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
