
#Carga y prepara los datos para el entrenamiento

import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

class DataLoader:
    #Gestiona la carga de imágenes desde carpetas"""
    
    def __init__(self, config):
        self.config = config
    
    def verify_data(self):
        #Verifica que las carpetas de datos existan y contengan imágenes
        
        #print("Verificando la estructura de datos")
        # Para clasificación binaria, solo necesitamos pet y stray
        required_classes = ['pet', 'stray']
        
        for class_name in required_classes:
            class_path = os.path.join(self.config.DATA_PATH, class_name)
            
            if not os.path.exists(class_path):
                raise FileNotFoundError(f"Carpeta no encontrada: {class_path}")
            
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"{class_name}: {len(images)} imágenes encontradas")
            
            if len(images) == 0:
                raise ValueError(f"No hay imágenes en: {class_path}")
        
        #print("Datos existentes \n")
    
    def load_data(self):
        #Carga y prepara los datos
        
        print("Cargando datos\n")
        
        images = []
        labels = []
        
        # Solo cargamos pet (0) y stray (1) para clasificación binaria
        class_mapping = {'pet': 0, 'stray': 1}
        
        for class_name, class_idx in class_mapping.items():
            class_path = os.path.join(self.config.DATA_PATH, class_name)
            
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Procesando {len(image_files)} imágenes de {class_name}...")
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                
                try:
                    img = load_img(img_path, target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE))
                    img_array = img_to_array(img)
                    img_array = img_array / 255.0  # Normalizar a [0, 1]
                    
                    images.append(img_array)
                    labels.append(class_idx)
                    
                except Exception as e:
                    print(f"Error cargando {img_file}: {e}")
            
            count = len([l for l in labels if l == class_idx])
            print(f"{class_name}: {count} imágenes cargadas")
        
        if len(images) == 0:
            raise Exception("No se pudieron cargar imágenes.")
        
        X = np.array(images)
        y = np.array(labels)
        
        # Dividir en entrenamiento y validación
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        print(f"\nDatos de entrenamiento: {len(X_train)} imágenes")
        print(f"Datos de validación: {len(X_test)} imágenes\n")
        
        return X_train, X_test, y_train, y_test