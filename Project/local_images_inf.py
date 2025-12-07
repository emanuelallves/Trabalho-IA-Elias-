# detect_yolo_opencv_folder.py

import os
import cv2
from ultralytics import YOLO

# --- 1. Carregar modelo YOLO ---
model_path = 'D:/TrabalhoIA/best.pt'
model = YOLO(model_path)

# --- 2. Pasta com as imagens ---
images_folder = 'D:/TrabalhoIA/kyara'  # pasta com as imagens

# Extensões válidas
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

# --- 3. Listar todas as imagens da pasta ---
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(valid_exts)]

if not image_files:
    raise ValueError("Nenhuma imagem encontrada na pasta")

# --- 4. Loop pelas imagens ---
for image_name in image_files:
    image_path = os.path.join(images_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Erro ao ler: {image_path}")
        continue

    # Fazer detecção
    results = model.predict(source=image, conf=0.4)

    # Desenhar bounding boxes
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            # Desenhar retângulo
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label
            label = f'{cls} {conf:.2f}'
            cv2.putText(
                image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

    # Mostrar nome do arquivo na janela
    scale = 0.4  # 60% do tamanho original
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)

    resized = cv2.resize(image, (width, height))
    cv2.imshow('Detections', resized)
    cv2.setWindowTitle('Detections', f'Detections - {image_name}')

    # Pressione qualquer tecla para ir pra próxima imagem
    key = cv2.waitKey(0)
    if key == 27:  # ESC para sair antes do fim
        break

cv2.destroyAllWindows()
