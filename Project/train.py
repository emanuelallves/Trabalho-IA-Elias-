from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    # Treinamento
    results = model.train(
        data="D:/TrabalhoIA/Project/config.yaml",  # caminho para o seu YAML
        epochs=100,
        imgsz=640,
        batch=2,
        name="dog_cat_detector",
        device=0  # GPU (0) ou 'cpu'
    )

if __name__ == "__main__":
    main()
