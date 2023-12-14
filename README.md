# Reconocimiento facial de emociones utilizando MobileViT
## 1. Introducción: 
En este proyecto, nos centramos en desarrollar un sistema de detección de emociones utilizando la arquitectura MobileViT. Este enfoque combina la eficiencia de MobileNet con la capacidad de atención de las transformers, lo que promete resultados precisos en la clasificación de expresiones faciales. A lo largo de este informe, desglosaremos cada componente esencial del proyecto para una comprensión más profunda. 

## 2. Preparación de Datos (cof_data.py): 
**Estructuración del Conjunto de Datos**

Organizamos las imágenes en carpetas de entrenamiento y prueba, organizando los datos. Esta estructura facilita la gestión y la carga de datos.

**Carga y Redimensionamiento de Imágenes**

Utilizamos OpenCV para cargar las imágenes en escala de grises y redimensionarlas a un tamaño uniforme. Esta uniformidad es esencial para optimizar la entrada del modelo y garantizar la coherencia en la información visual. 

**Almacenamiento Eficiente**

Los datos se almacenan en archivos. npz para un acceso rápido y eficiente durante el entrenamiento. Esta estrategia mejora la eficiencia y la velocidad de carga, especialmente cuando se trabaja con conjuntos de datos extensos. 

## 3. Aumentación de Datos (argumentation_data.py): 
**Enriquecimiento del Conjunto de Datos**

Aplicar volteo horizontal y vertical pretende proporcionar al modelo una mayor variabilidad en la orientación de las expresiones faciales. Este enfoque se basa en la lógica de que las emociones pueden ser expresadas de manera diferente según la posición del rostro, y al exponer al modelo a tales variaciones, se espera que mejore su capacidad para identificar estas emociones en diferentes contextos y ángulos. 

**Rotaciones y Zoom Controlado**

Las rotaciones y el zoom controlado agregan otro nivel de complejidad al conjunto de datos. Al introducir ligeras rotaciones, imitamos las variaciones naturales en la inclinación de las cabezas humanas, brindando al modelo una perspectiva más realista y robusta. Además, la aplicación de zoom controlado simula diferentes distancias entre la cámara y el sujeto, enriqueciendo aún más las características aprendidas por el modelo. 

**Mejora de la Generalización del Modelo**

Estas estrategias de aumentación trabajan en conjunto para mejorar la capacidad del modelo para generalizar a nuevas situaciones. Al exponerlo a una variedad más amplia de expresiones faciales y condiciones de imagen, buscamos aumentar su capacidad para reconocer y clasificar emociones de manera precisa y confiable en entornos del mundo real. 

 

## 4. Modelo MobileViT (mobilevit.py) 

**Definición de la Arquitectura**

La arquitectura MobileViT se define mediante TensorFlow y Keras. Incorpora bloques residuales invertidos y bloques transformers para aprovechar la eficiencia y capacidad de atención de esta arquitectura. 

**Bloques Residuales Invertidos:**
- Optimizan el entrenamiento con conexiones residuales. 
- Empiezan con convolución 1x1, seguida de 3x3. 
- Conexión residual para mantener información. 

 

**Bloques Transformers:** 
- Esenciales para procesamiento de secuencias y visión. 
- Atención multi-cabeza para relaciones a largo plazo. 
- Red MLP para aprendizaje no lineal. 
- Conexiones residuales y normalización 


**Variantes del Modelo**

Existen tres tamaños diferentes del modelo (MobileViT_S, MobileViT_XS, MobileViT_XXS), adaptándose así a distintos requisitos de recursos y complejidades de la tarea. 

**Compilación y Guardado del Modelo**

Compilamos el modelo con el optimizador Adam y la pérdida de entropía cruzada categórica. Posteriormente, guardamos el modelo en formato SavedModel para su despliegue y uso futuro. 

 

Para crear las metricas utilizamos lo siguiente :  
```
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras metrics.Recall()]) 
```
 

**Optimizer (Optimizador):** Se usa el optimizador 'adam', que ajusta la tasa de aprendizaje de manera dinámica durante el entrenamiento. 

**Loss Function (Función de Pérdida):** La función de pérdida 'categorical_crossentropy' mide la diferencia entre las predicciones y las etiquetas reales para entrenar al modelo en problemas de clasificación multiclase. 

**Metrics (Métricas):** Se seleccionan métricas para evaluar el rendimiento del modelo: 
- Accuracy mide la proporción de predicciones correctas. 

- Precision evalúa la precisión en clasificación positiva. 

- Recall mide la sensibilidad o tasa de verdaderos positivos. 

## 5. Entrenamiento del Modelo (train_model.py): 
**Carga de Datos** 

Cargamos los conjuntos de entrenamiento y prueba utilizando los archivos .npz previamente creados. 

**Normalización de Imágenes**

Normalizamos las imágenes para escalar los valores de píxeles en el rango [0, 1], facilitando la convergencia del modelo durante el entrenamiento. 

**Entrenamiento y Evaluación**

Entrenamos el modelo con el conjunto de entrenamiento y evaluamos su rendimiento en el conjunto de prueba. Monitorizamos resultados como pérdida y precisión durante este proceso. 

## 6. Detección de Emociones en Tiempo Real (emotion_classifier.py): 
**Integración de Modelo y Detección de Rostros**

Utilizamos la cámara web para capturar imágenes en tiempo real y aplicamos un clasificador Haar Cascade para detectar rostros. 

**Procesamiento de Imágenes de Rostros**

Procesamos las imágenes de rostros detectados mediante redimensionamiento y normalización antes de ser clasificadas por el modelo. 

**Visualización de Resultados**

Visualizamos los resultados de la clasificación en tiempo real, mostrando la emoción detectada y su probabilidad asociada para facilitar la interpretación del rendimiento del modelo. 
