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
Se usa ReLU como funcion de activacion por distintos motivos. En primer lugar porque las funciones Sigmoid y Tanh traen problemas al calcular el gradiente para actualizar los pesos, ya que la derivada da casi cero (Vanishing Gradient). La derivada de ReLU en cambio, da siempre 1 (para valores positivos). En segundo lugar ya que las funciones Sigmoid y Tanh podrian volver el proceso muy lento y pesado (al usar exponenciales complejas). 

**d. ¿Qué parámetro del modelo deberíamos cambiar si aumentamos el tamaño de entrada de la imagen?**
Si aumentamos el tamaño de la imagen hay que cambiar el imput size.

## 3. Entrenamiento y Optimización

**a. ¿Qué hace optimizer.zero_grad()?**
Se encarga de resetear y borrar los gradientes del entrenamiento anterior antes de calcular los nuevos. Si no ponemos optimizer.zero_grad(), PyTorch acumula y suma los gradientes de la época pasada con la nueva.

**b. ¿Por qué usamos CrossEntropyLoss() en este caso?**
La usamos porque estamos en un problema de clasificación multiclase (tenemos varias categorías de lesiones y cada foto pertenece a una sola). Esta función de pérdida es la ideal porque penaliza fuerte al modelo si está muy seguro de una respuesta incorrecta, ayudando a que aprenda a separar bien las clases durante el entrenamiento.

**c. ¿Cómo afecta la elección del tamaño de batch (batch_size) al entrenamiento?**
El batch size es un numero de imagenes que mira el modelo antes de calcular el error y actualizar los pesos. Si se elige un batch chico: el gradiente de saltos mas abruptos, por lo que el entrenamiento puede ser mas oscilante, pero generalizar mejor y evitar soluciones falsas. Ademas de que requiere menos memoria, pero puede ser mas lento. Por su parte en batchs mas grandes, el gradiente es mas estable. Curva de perdida mas suave, pero ojo con que el modelo memorice. Requiere mas memoria. 


**d. ¿Qué pasaría si no usamos model.eval() durante la validación?**
Al no poner model.eval en la validación, el modelo se quedaria en modo entrenamiento, con Dropout activo. Es decir se van a apagar neuronas al azar en VAL. Las metricas en VAL van a dar mucho peor de lo que deberian.

## 4. Validación y Evaluación

**a. ¿Qué significa una accuracy del 70% en validación pero 90% en entrenamiento?**
Esto significa Overfitting o sobreajuste. Se debe a que el modelo memorizo las fotos del conjunto de TRAIN, pero al validarlo con fotos nuevas la memorizacion ya no le sirve ya que no aprendio a distinguir (tan bien).

**b. ¿Qué otras métricas podrían ser más relevantes que accuracy en un problema real?**
El Recall y la Precision. El Accuracy te miente si tenés un dataset desbalanceado (muchas fotos sanas y pocas enfermas). El Recall es clave porque te dice cuántos enfermos reales detectaste (para no mandar a alguien enfermo a la casa), y la Precision te dice qué tan confiable sos cuando asegurás que una lesión es peligrosa.

**c. ¿Qué información útil nos da una matriz de confusión que no nos da la accuracy?**
El Accuracy te da un número global y te esconde los errores. La matriz de confusión te muestra el mapa completo de las pifias: te dice exactamente qué clase se está confundiendo con cuál.

**d. En el reporte de clasificación, ¿qué representan precision, recall y f1-score?**
Precision, indica de todo lo que el modelo dijo que era de una clase, cuánto le pegó de verdad (mide qué tan seguro es al tirar un resultado).
Recall, representat de todos los casos reales que había de esa clase, cuántos logró encontrar (mide que no se le escape ninguno).
F1-Score, el promedio entre las dos; te da una nota única para saber si el modelo está bien equilibrado entre Precision y Recall.

## 5. TensorBoard y Logging

**a. ¿Qué ventajas tiene usar TensorBoard durante el entrenamiento?**
Las ventajas de usar TensorBoard son que permite ver toda la información de forma visual mediante gráficos en lugar de mirar números sueltos en la consola, facilitando el monitoreo de las curvas de Loss y Accuracy en tiempo real. Esto ayuda a detectar problemas como el sobreajuste (overfitting) al instante al comparar Train y Validación en la misma pantalla. Además, permite contrastar visualmente el rendimiento de diferentes pruebas con distintos hiperparámetros en simultáneo.

**b. ¿Qué diferencias hay entre loguear add_scalar, add_image y add_text?**
La diferencia es lo que guardás: add_scalar es para números que cambian en cada época, como los gráficos de Loss y Accuracy. add_image es para mandar fotos directamente y ver qué está procesando la red. Y add_text es para guardar texto, ideal para dejar anotados los hiperparámetros que usaste en esa corrida.

**c. ¿Por qué es útil guardar visualmente las imágenes de validación en TensorBoard?**
Es útil para ver en qué le está pifiando el modelo en la vida real. Al mirar la foto junto con lo que predijo la red y la etiqueta real, te das cuenta al toque si se está confundiendo por culpa del brillo, el fondo o si alguna transformación rompió la imagen.


**d. ¿Cómo se puede comparar el desempeño de distintos experimentos en TensorBoard?**
Se comparan abriendo todos los experimentos juntos en la interfaz de TensorBoard. La herramienta les asigna un color diferente a cada corrida y te superpone las curvas de Loss y Accuracy en el mismo gráfico. Esto te permite activar o desactivar experimentos desde el panel lateral para contrastar visualmente y al toque cuál configuración de hiperparámetros funcionó mejor.

## 6. Generalización y Transferencia

**a. ¿Qué cambios habría que hacer si quisiéramos aplicar este mismo modelo a un dataset con 100 clases?**
Hay que cambiar el parámetro num_classes = 100 para agrandar la última capa de la red. El modelo que tenemos ahora está configurado para dar menos respuestas; si no le ponemos 100 neuronas a la salida, el código va a tirar error de dimensiones porque no va a coincidir con las 100 etiquetas nuevas del dataset

**b. ¿Por qué una CNN suele ser más adecuada que una MLP para clasificación de imágenes?**
Porque la CNN mantiene la forma de la foto (mira los píxeles vecinos en 2D como con una lupa), lo que le permite entender bordes y texturas sin importar dónde estén. La MLP es más bruta: nos obliga a 'aplanar' la foto en una sola fila gigante, haciendo que se pierda la relación del espacio y que la cantidad de parámetros explote a millones, haciendo que se memorice las fotos de memoria.

**c. ¿Qué problema podríamos tener si entrenamos este modelo con muy pocas imágenes por clase?**
El problema es que vamos a caer de cabeza en Overfitting o sobreajuste. Como la red es grande y tiene muy pocas fotos para practicar, le va a resultar facil memorizarse los detalles de esas pocas fotos, en vez de aprender a distinguir de verdad. Va a dar un Accuracy altísimo en TRAIN pero no en VAL.

**d. ¿Cómo podríamos adaptar este pipeline para imágenes en escala de grises?**
Habria que modificar la carga de datos, configurarando las transformaciones para que lean las fotos en 1 solo canal de color en vez de 3 (RGB).
Y el input_size, hay que sacar el * 3 de la primera capa del modelo.

## 7. Regularización

### Preguntas teóricas:
**a. ¿Qué es la regularización en el contexto del entrenamiento de redes neuronales?**
**b. ¿Cuál es la diferencia entre `Dropout` y regularización `L2` (weight decay)?**
**c. ¿Qué es `BatchNorm` y cómo ayuda a estabilizar el entrenamiento?**
**d. ¿Cómo se relaciona `BatchNorm` con la velocidad de convergencia?**
**e. ¿Puede `BatchNorm` actuar como regularizador? ¿Por qué?**
**f. ¿Qué efectos visuales podrías observar en TensorBoard si hay overfitting?**
**g ¿Cómo ayuda la regularización a mejorar la generalización del modelo?**

### Actividades de modificación:
1. Agregar Dropout en la arquitectura MLP:
   - Insertar capas `nn.Dropout(p=0.5)` entre las capas lineales y activaciones.
   - Comparar los resultados con y sin `Dropout`.

2. Agregar Batch Normalization:
   - Insertar `nn.BatchNorm1d(...)` después de cada capa `Linear` y antes de la activación:
     ```python
     self.net = nn.Sequential(
         nn.Flatten(),
         nn.Linear(in_features, 512),
         nn.BatchNorm1d(512),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(512, 256),
         nn.BatchNorm1d(256),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(256, num_classes)
     )
     ```

3. Aplicar Weight Decay (L2):
   - Modificar el optimizador:
     ```python
     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
     ```

4. Reducir overfitting con data augmentation:
   - Agregar transformaciones en Albumentations como `HorizontalFlip`, `BrightnessContrast`, `ShiftScaleRotate`.

5. Early Stopping (opcional):
   - Implementar un criterio para detener el entrenamiento si la validación no mejora después de N épocas.

### Preguntas prácticas:
**a. ¿Qué efecto tuvo `BatchNorm` en la estabilidad y velocidad del entrenamiento?**
**b. ¿Cambió la performance de validación al combinar `BatchNorm` con `Dropout`?**
**c. ¿Qué combinación de regularizadores dio mejores resultados en tus pruebas?**
**d. ¿Notaste cambios en la loss de entrenamiento al usar `BatchNorm`?**

## 8. Inicialización de Parámetros

### Preguntas teóricas:
**a. ¿Por qué es importante la inicialización de los pesos en una red neuronal?**
**b. ¿Qué podría ocurrir si todos los pesos se inicializan con el mismo valor?**
**c. ¿Cuál es la diferencia entre las inicializaciones de Xavier (Glorot) y He?**
**d. ¿Por qué en una red con ReLU suele usarse la inicialización de He?**
**e. ¿Qué capas de una red requieren inicialización explícita y cuáles no?**

### Actividades de modificación:
1. Agregar inicialización manual en el modelo:
   - En la clase `MLP`, agregar un método `init_weights` que inicialice cada capa:
     ```python
     def init_weights(self):
         for m in self.modules():
             if isinstance(m, nn.Linear):
                 nn.init.kaiming_normal_(m.weight)
                 nn.init.zeros_(m.bias)
     ```

2. Probar distintas estrategias de inicialización:
   - Xavier (`nn.init.xavier_uniform_`)
   - He (`nn.init.kaiming_normal_`)
   - Aleatoria uniforme (`nn.init.uniform_`)
   - Comparar la estabilidad y velocidad del entrenamiento.

3. Visualizar pesos en TensorBoard:
   - Agregar esta línea en la primera época para observar los histogramas:
     ```python
     for name, param in model.named_parameters():
         writer.add_histogram(name, param, epoch)
     ```

### Preguntas prácticas:
**a. ¿Qué diferencias notaste en la convergencia del modelo según la inicialización?**
**b. ¿Alguna inicialización provocó inestabilidad (pérdida muy alta o NaNs)?**
**c. ¿Qué impacto tiene la inicialización sobre las métricas de validación?**
**d. ¿Por qué `bias` se suele inicializar en cero?**