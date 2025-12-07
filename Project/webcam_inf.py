# detect_yolo_opencv_webcam_20frames.py

import cv2
from ultralytics import YOLO

# --- 1. Carregar modelo ---
model_path = 'D:/TrabalhoIA/best.pt'
model = YOLO(model_path)

# --- 2. Webcam ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Não foi possível acessar a webcam")

print("Rodando... Pressione 'q' para sair")

frame_count = 0
skip = 20  # a cada 20 frames faz a predição

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Só roda o modelo a cada 20 frames
    if frame_count % skip == 0:
        results = model.predict(source=frame, conf=0.4, verbose=False)

        # Desenhar boxes SOMENTE nesses frames
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{cls} {conf:.2f}'
                cv2.putText(
                    frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

    # Redimensionar para exibição
    scale = 0.4
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    resized = cv2.resize(frame, (width, height))

    cv2.imshow('Detections - Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
