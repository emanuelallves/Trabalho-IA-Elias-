# detect_yolo_opencv_folder_save.py

import os
import cv2
from ultralytics import YOLO

# --- 1. Carregar modelo YOLO ---
model_path = 'D:/TrabalhoIA/best.pt'
model = YOLO(model_path)

# --- 2. Pastas ---
images_folder = 'D:/TrabalhoIA/kyara'
output_folder = 'D:/TrabalhoIA/kyara_pred'

# Criar pasta de saída se não existir
os.makedirs(output_folder, exist_ok=True)

# Extensões válidas
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

# --- 3. Listar imagens ---
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(valid_exts)]

if not image_files:
    raise ValueError("Nenhuma imagem encontrada na pasta")

# --- 4. Processar imagens ---
for image_name in image_files:
    image_path = os.path.join(images_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Erro ao ler: {image_path}")
        continue

    # Fazer deteção
    results = model.predict(source=image, conf=0.4, verbose=False)

    # Desenhar bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{cls} {conf:.2f}'
            cv2.putText(
                image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

    # --- 5. Salvar imagem com predições ---
    output_path = os.path.join(output_folder, f"pred_{image_name}")
    cv2.imwrite(output_path, image)

    # --- 6. Mostrar na tela (opcional) ---
    scale = 0.4
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    resized = cv2.resize(image, (width, height))

    cv2.imshow('Detections', resized)
    cv2.setWindowTitle('Detections', f'Detections - {image_name}')

    # Pressione qualquer tecla para próxima imagem
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()
