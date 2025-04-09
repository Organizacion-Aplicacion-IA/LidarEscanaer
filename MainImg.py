import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Configuración
MODEL_PATH = "best_saved_model/best_float32.tflite"
IMAGE_PATH = "caja.jpeg"
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5

# Cargar modelo TFLite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Detalles de entrada/salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Cargar y preprocesar imagen
img = cv2.imread(IMAGE_PATH)
img_height, img_width = img.shape[:2]
img_resized = cv2.resize(img, (640, 640))
img_normalized = img_resized / 255.0
input_data = np.expand_dims(img_normalized, axis=0).astype(np.float32)

# Inferencia
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Obtener resultados
outputs = interpreter.get_tensor(output_details[0]['index'])[0]

# Decodificar detecciones
boxes = []
confidences = []

for i in range(outputs.shape[1]):
    x_center, y_center, width, height, confidence = outputs[:, i]
    
    if confidence > CONF_THRESHOLD:
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)
        
        boxes.append([x1, y1, x2, y2])
        confidences.append(float(confidence))

# NMS
indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

# Dibujar resultados y recortar objetos
for idx, i in enumerate(indices):
    i = i[0] if isinstance(i, (list, np.ndarray)) else i  # por si es array
    x1, y1, x2, y2 = boxes[i]

    # Dibujar caja en la imagen original
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"Caja_Clip: {confidences[i]:.2f}", 
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Recortar y guardar la imagen del objeto detectado
    cropped_obj = img[y1:y2, x1:x2]
    if cropped_obj.size > 0:
        output_name = f"objeto_detectado_{idx+1}.jpg"
        cv2.imwrite(output_name, cropped_obj)
        print(f"Objeto guardado: {output_name}")

# Mostrar imagen original con detecciones
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

# Guardar imagen final con cajas
plt.imsave("resultado_completo.jpg", img_rgb)
