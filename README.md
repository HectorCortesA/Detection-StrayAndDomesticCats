# Detection-StrayAndDomesticCats
Detection of stray and domestic cats with neural network, CNN.

El modelo utiliza una arquitectura CNN profunda con 4 capas convolucionales, normalización por lotes (BatchNormalization), pooling y dropout para prevenir el sobreajuste. El modelo está diseñado para aprender características distintivas entre gatos domésticos y callejeros a partir de imágenes.

Estructura del Modelo
Capas Convolucionales
Capa 1:

Filtros convolucionales con activación ReLU

BatchNormalization (acelera el entrenamiento reduciendo el "covariate shift")

MaxPooling2D (reduce dimensiones: 224→112)

Dropout (apaga neuronas aleatoriamente)

Capas 2, 3 y 4:

Patrón similar a Capa 1 con reducciones sucesivas: 112→56→28→14

Capa Fully Connected
Flatten: Convierte la salida 3D a 1D (14×14×256 = 50,176 valores)

BatchNormalization

Dropout

Capa Dense de salida:

1 neurona con activación sigmoide para clasificación binaria (0=gato doméstico, 1=gato callejero)

Configuración del Entrenamiento
Compilación
Función de pérdida: binary_crossentropy

Optimizador: Adam

Métrica: accuracy (exactitud)

Entrenamiento
División de datos: 80% entrenamiento, 20% validación

Data Augmentation: Genera variaciones de imágenes para mejorar generalización

Callbacks:

Guarda el mejor modelo

Early stopping (detiene entrenamiento sin mejora)

Reduce tasa de aprendizaje dinámicamente

Dataset
El modelo fue entrenado con el dataset: Pet Cats and Stray Cats 2025

Características del Dataset:
Imágenes de gatos domésticos y callejeros

Balanceado entre las dos clases

Imágenes preprocesadas a dimensiones uniformes

Uso
Clona el repositorio

Instala las dependencias: pip install -r requirements.txt

Prepara el dataset en la estructura de directorios correcta

Ejecuta el script de entrenamiento: python train.py

Resultados
El modelo logra una alta precisión en la clasificación binaria, demostrando efectividad en distinguir características entre gatos domésticos y callejeros basándose en patrones visuales aprendidos durante el entrenamiento.

Tecnologías Utilizadas
Python

TensorFlow/Keras

OpenCV (para preprocesamiento)

NumPy, Pandas

Scikit-learn

Autor
Desarrollado como proyecto de aprendizaje automático para clasificación de imágenes.

Licencia
