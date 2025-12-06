
from tensorflow import keras
import json
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from config import Config


class Predictor:
    #Realiza predicciones en nuevas imágenes
    
    def __init__(self, config):
        self.config = config
        self.model = None
    
    def load_model(self):
        #Carga modelo entrenado
        
        model_path = os.path.join(self.config.MODELS_PATH, self.config.MODEL_NAME)
        
        if not os.path.exists(model_path):
            print(f"Error: Modelo no encontrado en {model_path}")
            return False
        
        self.model = keras.models.load_model(model_path)
        print(f"Modelo cargado desde: {model_path}\n")
        return True
    
    def predict_single_image(self, image_path):
    #Realiza predicción en una sola imagen
        
        if not os.path.exists(image_path):
            print(f"Error: Imagen no encontrada en {image_path}")
            return None
        
        if self.model is None:
            print("Error: Modelo no cargado")
            return None
        
        try:
            img = load_img(image_path, target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE))
            img_array = img_to_array(img) / 255.0
            
            prediction = self.model.predict(np.array([img_array]), verbose=0)[0][0]
            
            class_idx = 1 if prediction > 0.5 else 0
            classification = self.config.CLASS_NAMES[class_idx]
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            result = {
                'image': os.path.basename(image_path),
                'classification': classification,
                'confidence': float(confidence),
                'raw_probability': float(prediction)
            }
            
            return result
        
        except Exception as e:
            print(f"Error al procesar imagen: {e}")
            return None
    
    def predict_batch(self, folder_path):
        #Realiza predicciones en múltiples imágenes de una carpeta#
        
        if not os.path.isdir(folder_path):
            print(f"Error: Carpeta no encontrada {folder_path}")
            return []
        
        results = []
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Prediciendo en {len(image_files)} imágenes...\n")
        
        for idx, filename in enumerate(image_files, 1):
            image_path = os.path.join(folder_path, filename)
            result = self.predict_single_image(image_path)
            
            if result:
                results.append(result)
                print(f"[{idx}/{len(image_files)}] {filename}")
                print(f" Clasificación: {result['classification']}")
                print(f" Confianza: {result['confidence']:.2%}\n")
        
        return results
    
    def print_prediction(self, result):
        #Imprime predicción de manera legible
        
        if result is None:
            return
        
        print("RESULTADO DE PREDICCIÓN")
        print(f"Imagen: {result['image']}")
        print(f"Clasificación: {result['classification']}")
        print(f"Confianza: {result['confidence']:.2%}")
        print(f"Probabilidad (raw): {result['raw_probability']:.4f}")
        print("="*60 + "\n")
    
    def save_results(self, results, output_file):
        #Guarda resultados en archivo JSON
        
        output_path = os.path.join(self.config.RESULTS_PATH, output_file)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Resultados guardados en: {output_path}\n")

def predict_image(image_path):
    #Función para predecir una imagen
    
    config = Config()
    predictor = Predictor(config)
    
    if not predictor.load_model():
        return None
    
    result = predictor.predict_single_image(image_path)
    predictor.print_prediction(result)
    
    return result

def predict_folder(folder_path):
    #Función para predecir carpeta completa
    
    config = Config()
    predictor = Predictor(config)
    
    if not predictor.load_model():
        return None
    
    results = predictor.predict_batch(folder_path)
    predictor.save_results(results, 'predictions.json')
    
    # Resumen
    mascotas = sum(1 for r in results if r['classification'] == 'Gato Mascota')
    callejeros = sum(1 for r in results if r['classification'] == 'Gato Callejero')
    
    print("\n" + "-"*60)
    print("RESUMEN")
    print(f"Total predicciones: {len(results)}")
    print(f"Gatos Mascota: {mascotas}")
    print(f"Gatos Callejeros: {callejeros}")
    
    return results
