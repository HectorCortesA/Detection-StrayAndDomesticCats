"""
config.py - Configuración del proyecto CNN para clasificación de gatos
"""

import os
import json

class Config:
    """Clase de configuración centralizada"""
    
    # Configuración de datos
    DATA_PATH = './data'
    MODELS_PATH = './models'
    RESULTS_PATH = './results'
    
    # Configuración del modelo
    IMG_SIZE = 224
    IMG_CHANNELS = 3
    BATCH_SIZE = 32
    EPOCHS = 50
    
    # Clases para clasificación binaria (SOLO PET y STRAY)
    CLASSES = {
        'pet': 0,           # Gatos Mascota
        'stray': 1          # Gatos Callejeros
    }
    
    # Nombres de clases para mostrar
    CLASS_NAMES = {
        0: 'Gato Mascota',
        1: 'Gato Callejero'
    }
    
    # Modelo
    MODEL_NAME = 'cat_classifier_cnn.h5'
    TRAINING_INFO_FILE = 'training_info.json'
    
    def __init__(self):
        """Inicializa y crea directorios necesarios"""
        self._create_directories()
        self.training_info = self._load_training_info()
    
    def _create_directories(self):
        """Crea los directorios necesarios para el proyecto"""
        directories = [self.DATA_PATH, self.MODELS_PATH, self.RESULTS_PATH]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"[CONFIG] Directorio creado: {directory}")
    
    def _load_training_info(self):
        """Carga la información del entrenamiento previo"""
        info_path = os.path.join(self.RESULTS_PATH, self.TRAINING_INFO_FILE)
        
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[CONFIG] Error cargando training_info: {e}")
                return {'is_trained': False, 'epochs_completed': 0}
        
        return {'is_trained': False, 'epochs_completed': 0}
    
    def save_training_info(self, is_trained=True, epochs_completed=0):
        """Guarda la información del entrenamiento"""
        self.training_info = {
            'is_trained': is_trained,
            'epochs_completed': epochs_completed,
            'model_name': self.MODEL_NAME,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        info_path = os.path.join(self.RESULTS_PATH, self.TRAINING_INFO_FILE)
        
        try:
            with open(info_path, 'w') as f:
                json.dump(self.training_info, f, indent=4)
            return True
        except Exception as e:
            print(f"[CONFIG] Error guardando training_info: {e}")
            return False
    
    def is_model_trained(self):
        """Verifica si ya hay un modelo entrenado"""
        # Verificar ambos formatos
        model_path_h5 = os.path.join(self.MODELS_PATH, self.MODEL_NAME)
        model_path_keras = os.path.join(self.MODELS_PATH, 'cat_classifier_cnn.keras')
        
        model_exists = os.path.exists(model_path_h5) or os.path.exists(model_path_keras)
        
        # También verificar en la info
        info_trained = self.training_info.get('is_trained', False)
        
        return model_exists and info_trained
    
    def get_model_path(self):
        """Obtiene la ruta del modelo (prioriza .h5)"""
        model_path_h5 = os.path.join(self.MODELS_PATH, self.MODEL_NAME)
        model_path_keras = os.path.join(self.MODELS_PATH, 'cat_classifier_cnn.keras')
        
        if os.path.exists(model_path_h5):
            return model_path_h5
        elif os.path.exists(model_path_keras):
            return model_path_keras
        else:
            return None

# Importar time para timestamp
import time