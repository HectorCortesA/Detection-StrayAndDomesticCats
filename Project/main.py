# Ejecutar proyecto CNN de clasificación de gatos
import sys
import os
from config import Config
from train import train_model
from evaluate import evaluate_model
from predict import predict_image, predict_folder
from tensorflow.keras.models import load_model
import tensorflow as tf



def print_menu(config, model_loaded):
  #  """Imprime el menú con estado actual"""
    print("-"*70)
    
    model_trained = config.is_model_trained()
    model_in_memory = model_loaded is not None
    
    if model_trained:
        print(" Modelo entrenado disponible en disco")
        epochs = config.training_info.get('epochs_completed', 0)
        print(f"  Épocas entrenadas: {epochs}")
    
    if model_in_memory:
        print(" Modelo cargado en memoria")
    
    print("-"*70)
    
    print("\n1. Entrenar modelo desde cero")
    print("2. Evaluar modelo actual")
    print("3. Predecir imagen individual")
    print("4. Predecir carpeta completa")
    print("5. Cargar modelo (si existe)")
    print("6. Estado del sistema")
    print("7. Salir")

def load_model_from_disk():
  #  """Carga el modelo desde disco si existe"""
    config = Config()
    
    if not config.is_model_trained():
        print("\nError: No hay modelo entrenado disponible en disco")
        return None
    
    model_path = os.path.join(config.MODELS_PATH, config.MODEL_NAME)
    
    try:
        # Usar el nuevo formato .keras si existe
        keras_model_path = os.path.join(config.MODELS_PATH, 'cat_classifier_cnn.keras')
        if os.path.exists(keras_model_path):
            model_path = keras_model_path
        
        print(f"\nCargando modelo desde: {model_path}")
        model = load_model(model_path)
        print(f"Modelo cargado exitosamente")
        return model
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        
        # Intentar cargar con formato HDF5 si falla
        try:
            print("Intentando cargar en formato HDF5...")
            model = tf.keras.models.load_model(model_path)
            print("Modelo cargado en formato HDF5")
            return model
        except Exception as e2:
            print(f" Error también en formato HDF5: {e2}")
            return None

def check_system_status():
  #  """Muestra el estado actual del sistema"""
    config = Config()
    
    print("\n" + "-"*70)
    print("ESTADO DEL SISTEMA")
    
    # Verificar datos
    print("\nDATOS:")
    data_dirs = ['pet', 'stray']
    for dir_name in data_dirs:
        dir_path = os.path.join(config.DATA_PATH, dir_name)
        if os.path.exists(dir_path):
            images = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  {dir_name}/: {len(images)} imágenes")
        else:
            print(f"  {dir_name}/: NO ENCONTRADO")
    
    # Verificar modelo
    print("\nMODELO:")
    model_path = os.path.join(config.MODELS_PATH, config.MODEL_NAME)
    model_exists = os.path.exists(model_path)
    
    # También verificar formato .keras
    keras_model_path = os.path.join(config.MODELS_PATH, 'cat_classifier_cnn.keras')
    keras_model_exists = os.path.exists(keras_model_path)
    
    if model_exists or keras_model_exists:
        actual_path = keras_model_path if keras_model_exists else model_path
        print(f"  ✓ Modelo encontrado: {os.path.basename(actual_path)}")
        print(f"  Tamaño: {os.path.getsize(actual_path) / 1024 / 1024:.2f} MB")
        
        if config.training_info.get('is_trained', False):
            epochs = config.training_info.get('epochs_completed', 0)
            print(f"  Épocas entrenadas: {epochs}")
        else:
            print("Información de entrenamiento no encontrada")
    else:
        print("Modelo NO encontrado")
    
    # Verificar resultados
    print("\nRESULTADOS:")
    results_files = ['training_history.pkl', 'training_info.json']
    for file_name in results_files:
        file_path = os.path.join(config.RESULTS_PATH, file_name)
        if os.path.exists(file_path):
            print(f"{file_name}")
        else:
            print(f"{file_name} (no encontrado)")

def main():
   # """Función principal"""
    config = Config()
    
    model = None  # Modelo en memoria
    X_test = None
    y_test = None
    
    while True:
        print_menu(config, model)
        
        choice = input("\nSelecciona una opción (1-7): ").strip()
        
        if choice == '1':
            print("\n" + "-"*70)
            # Preguntar confirmación si ya hay modelo
            if config.is_model_trained():
                confirm = input("Ya existe un modelo entrenado. ¿Sobrescribir? (sí/no): ").strip().lower()
                if confirm not in ['si', 'sí', 's', 'yes', 'y']:
                    print("[MAIN] Entrenamiento cancelado\n")
                    continue
            
            print("Iniciando entrenamiento desde cero...\n")
            try:
                model, X_test, y_test = train_model()
                print("Entrenamiento desde cero completado")
                
                # Actualizar config después de entrenar
                config = Config()
            except Exception as e:
                print(f"[MAIN] ✗ Error durante entrenamiento: {e}")
        
        elif choice == '2':
            print("\n" + "-"*70)
            print("EVALUACIÓN DEL MODELO")
            
            # Verificar si el modelo está en memoria, si no cargarlo
            if model is None:
                print("Modelo no cargado en memoria, intentando cargar desde disco...")
                model = load_model_from_disk()
            
            if model is None:
                print("\n Error: No se pudo cargar el modelo")
                print(" Opciones:")
                print("  1. Entrene un modelo primero (opción 1)")
                print("  2. Cargue un modelo existente (opción 5)\n")
                continue
            
            # Cargar datos para evaluación
            print("Cargando datos para evaluación...")
            try:
                from data_loader import DataLoader
                data_loader = DataLoader(config)
                data_loader.verify_data()
                _, X_test, _, y_test = data_loader.load_data()
                
                print(f"Datos cargados: {len(X_test)} imágenes de prueba\n")
                
                print("Iniciando evaluación...\n")
                evaluate_model(model, X_test, y_test)
                print("Evaluación completada exitosamente")
                
            except Exception as e:
                print(f" Error durante evaluación: {e}")
                # Reiniciar variables en caso de error
                X_test = None
                y_test = None
        
        elif choice == '3':
            if model is None:
                print("\nModelo no cargado, intentando cargar desde disco...")
                model = load_model_from_disk()
            
            if model is None:
                print("\nError: No se pudo cargar el modelo")
                print("Entrene o cargue un modelo primero\n")
                continue
            print("PREDICCIÓN DE IMAGEN INDIVIDUAL")
            
            image_path = input("Ingresa la ruta de la imagen: ").strip()
            
            if not os.path.exists(image_path):
                print(f"Error: Archivo no encontrado: {image_path}\n")
            else:
                try:
                    predict_image(image_path)
                except Exception as e:
                    print(f"Error: {e}\n")
        
        elif choice == '4':
            if model is None:
                print("\nModelo no cargado, intentando cargar desde disco...")
                model = load_model_from_disk()
            
            if model is None:
                print("\nError: No se pudo cargar el modelo")
                print("Entrene o cargue un modelo primero\n")
                continue
            
            print("PREDICCIÓN DE CARPETA COMPLETA")
            
            folder_path = input("Ingresa la ruta de la carpeta: ").strip()
            
            if not os.path.isdir(folder_path):
                print(f" Error: Carpeta no encontrada: {folder_path}\n")
            else:
                try:
                    predict_folder(folder_path)
                except Exception as e:
                    print(f"Error: {e}\n")
        
        elif choice == '5':
            print("\n" + "-"*70)
            print("CARGA DE MODELO")
            
            if config.is_model_trained():
                model = load_model_from_disk()
                if model is not None:
                    print("\nModelo listo para evaluación o predicción")
                    print("Ahora puede usar las opciones 2, 3 o 4\n")
            else:
                print(" No hay modelo entrenado disponible en disco")
                print(" Use la opción 1 para entrenar un modelo\n")
        
        elif choice == '6':
            check_system_status()
        
        elif choice == '7':
            print("\nHasta luego\n")
            break
        
        else:
            print("\n Opción no válida\n")

if __name__ == "__main__":
    try:
        main()
    
    except KeyboardInterrupt:
        print("\n Programa interrumpido por el usuario\n")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n Error inesperado: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)