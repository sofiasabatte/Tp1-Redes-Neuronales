# PREGUNTAS SOBRE EL EJEMPLO DE CLASIFICACIÓN DE IMAGENES CON PYTORCH Y MLP

## 1. Dataset y Preprocesamiento

**a. ¿Por qué es necesario redimensionar las imágenes a un tamaño fijo para una MLP?**
La MLP (Red Neuronal Multicapa), tiene una cantidad fija de entradas. Es por esto que se deben redimensionar todas las imaganes al mismo tamaño, para que tengan la misma cantidad de pixeles, y por ende la misma cantidad de numeros al hacer el flatten.

**b. ¿Qué ventajas ofrece Albumentations frente a otras librerías de transformación como torchvision.transforms?**
Son librerías que sirven para transformar imágenes, tanto para preprocesarlas (redimensionar, normalizar) como para hacer data augmentation, generar versiones modificadas de las imágenes  para auemntar tu dataset para el entrenmiento. Albumentations tiene ventajas frente a torchvision: es más rápida, tiene más variedad de transformaciones y permite aplicarlas con probabilidades distintas para cada una.

**c. ¿Qué hace A.Normalize()? ¿Por qué es importante antes de entrenar una red?**
Las imagenes tienen pixeles entre 0 y 255. A.Normalize() busca transformar estos valores usando medio 0 y desvio estandar 1, para achicar la escala y volver el dataset mas uniforme. Esto es importantisimo para lograr un entrenamiento mas rspido y estable.

**d. ¿Por qué convertimos las imágenes a ToTensorV2() al final de la pipeline?**
ToTensorV2() busca convertir el datset a un formato que entiende la red. PyTorch no entiende imagenes sino que trabaja con tensores que son arrays multidimensionales de números que la GPU puede procesar eficientemente. La imagen entra como un array de numpy con forma (alto, ancho, caneles); y ToTensorV2() la convierte a un tensor de PyTorch cambiando el orden de las dimensiones a (canales, alto, ancho), que es el formato que espera la red.

## 2. Arquitectura del Modelo

**a. ¿Por qué usamos una red MLP en lugar de una CNN aquí? ¿Qué limitaciones tiene?**
Una MLP, es una red neuronal donde cada neurona esta conectada con todas las neuronas de la siguiente capa. La entrada es el tensor mencionado anteriormente (la imagen con el flatten). Por su parte las CNN (Redes Neuronales Convolucionales), estan diseñadas para procesar la imagen por trozos para encontrar patrones. 
Usamos MLP aca,  porque es un modelo mas basico para arrancar a entender como funcionan las redes nueronales. Las desventajas que va a traer van a ser: al hacer un flatten, se va a perder la estructura espacial (entender si algo es una mancha o una textura por ejemplo); ademas tiene muchos parametros, volviendo el proceso lento y propenso a errores (como overfitting); y ademas, no detecta patrones locales.

**b. ¿Qué hace la capa Flatten() al principio de la red?**
La capa flatten mencionada ya anteriormente, lo que hace es "desenrollar" la imagen, multiplicando ancho alto y canales para que la MLP la pueda procesar.

**c. ¿Qué función de activación se usó? ¿Por qué no usamos Sigmoid o Tanh?**

**d. ¿Qué parámetro del modelo deberíamos cambiar si aumentamos el tamaño de entrada de la imagen?**

## 3. Entrenamiento y Optimización

**a. ¿Qué hace optimizer.zero_grad()?**

**b. ¿Por qué usamos CrossEntropyLoss() en este caso?**

**c. ¿Cómo afecta la elección del tamaño de batch (batch_size) al entrenamiento?**

**d. ¿Qué pasaría si no usamos model.eval() durante la validación?**

## 4. Validación y Evaluación

**a. ¿Qué significa una accuracy del 70% en validación pero 90% en entrenamiento?**

**b. ¿Qué otras métricas podrían ser más relevantes que accuracy en un problema real?**

**c. ¿Qué información útil nos da una matriz de confusión que no nos da la accuracy?**

**d. En el reporte de clasificación, ¿qué representan precision, recall y f1-score?**

## 5. TensorBoard y Logging

**a. ¿Qué ventajas tiene usar TensorBoard durante el entrenamiento?**

**b. ¿Qué diferencias hay entre loguear add_scalar, add_image y add_text?**

**c. ¿Por qué es útil guardar visualmente las imágenes de validación en TensorBoard?**

**d. ¿Cómo se puede comparar el desempeño de distintos experimentos en TensorBoard?**

## 6. Generalización y Transferencia

**a. ¿Qué cambios habría que hacer si quisiéramos aplicar este mismo modelo a un dataset con 100 clases?**

**b. ¿Por qué una CNN suele ser más adecuada que una MLP para clasificación de imágenes?**

**c. ¿Qué problema podríamos tener si entrenamos este modelo con muy pocas imágenes por clase?**

**d. ¿Cómo podríamos adaptar este pipeline para imágenes en escala de grises?**

