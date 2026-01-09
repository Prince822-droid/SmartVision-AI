import cv2
import numpy as np
import os
import traceback

# ---------------- BASIC INFO ----------------
print("Current folder:", os.getcwd())
print("OpenCV version:", cv2.__version__)

# ---------------- FILE PATHS ----------------
weights = r"C:\Users\Prince_A14\Desktop\smartVisionAIrobot\yolov3-tiny.weights"
cfg = r"C:\Users\Prince_A14\Desktop\smartVisionAIrobot\yolov3-tiny.cfg"
names_path = r"C:\Users\Prince_A14\Desktop\smartVisionAIrobot\coco.names"

for p in (weights, cfg, names_path):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required file not found: {p}")

# ---------------- LOAD MODEL ----------------
net = cv2.dnn.readNet(weights, cfg)

# ---------------- LOAD CLASSES ----------------
with open(names_path, "r") as f:
    classes = [c.strip() for c in f.readlines() if c.strip()]

layer_names = net.getLayerNames()
outs_ids = net.getUnconnectedOutLayers()
outs_ids = [int(i) for i in np.array(outs_ids).reshape(-1)]
output_layers = [layer_names[i - 1] for i in outs_ids]
print("Output layers:", output_layers)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
INPUT_SIZE = (416, 416)

# ---------------- MAIN LOOP ----------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed")
            break

        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, INPUT_SIZE, (0, 0, 0),
            swapRB=True, crop=False
        )
        net.setInput(blob)

        try:
            outs = net.forward(output_layers)
        except Exception:
            traceback.print_exc()
            break

        boxes = []
        confidences = []
        class_ids = []

        # flags reset every frame
        person_detected = False
        phone_detected = False

        for out in outs:
            for detection in out:
                scores = detection[5:]
                if len(scores) == 0:
                    continue

                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence > CONF_THRESHOLD:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD
        )
        indices = np.array(indices).reshape(-1) if len(indices) > 0 else []

        # ---------------- DRAW & DETECT ----------------
        for i in indices:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            conf = confidences[i]

            if label == "person":
                person_detected = True
            if label == "cell phone":
                phone_detected = True

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(
                frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # ---------------- DECISION ----------------
        if person_detected:
            print("🛑 PERSON DETECTED → STOP")
        elif phone_detected:
            print("📱 CELL PHONE DETECTED → ALERT")
        else:
            print("🔍 SEARCHING...")

        # ---------------- DISPLAY ----------------
        cv2.imshow("YOLOv3-TinyObject Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()