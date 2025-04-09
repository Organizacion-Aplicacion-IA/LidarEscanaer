import cv2
from ultralytics import YOLO

# Cargar el modelo YOLOv8 (asegurándote de que el archivo .pt de tu modelo esté disponible)
model = YOLO('best.pt') 

# Inicializar la cámara web
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

while True:
    # Capturar fotogramas desde la cámara
    ret, frame = cap.read()
    
    if not ret:
        print("No se pudo recibir el marco (stream terminado?). Saliendo ...")
        break
    
    # Realizar la inferencia con El modelo
    results = model.predict(frame, imgsz=640, conf=0.90)
    

    frame_with_boxes = results[0].plot() 
    
  
    cv2.imshow('Detección en vivo', frame_with_boxes)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
