
#Entrenamiento del modelo CNN
import pickle
import time
import os
import numpy as np
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from config import Config
from data_loader import DataLoader
from model_builder import ModelBuilder
    
    
    #Gestiona el entrenamiento del modelo
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.history = None #Historial de entrenamiento
    
    def train(self, model, X_train, X_test, y_train, y_test):
        #Entrena el modelo desde cero
        
        print("-"*70)
        print("INICIANDO ENTRENAMIENTO")
        
        # Callbacks para mejorar entrenamiento
        checkpoint_path = os.path.join(self.config.MODELS_PATH, 'best_model.h5')
        
        callbacks = [
            ModelCheckpoint( # Guardar el mejor modelo
                filepath=checkpoint_path,
                monitor='val_accuracy', # Mejorar accuracy en validación
                save_best_only=True, # Guardar solo el mejor modelo
                mode='max', # Maximizar accuracy
                verbose=1 # Mostrar mensajes
            ),
            EarlyStopping( # Detener si no hay mejora
                monitor='val_loss',
                patience=20, # Esperar 20 épocas sin mejora
                restore_best_weights=True, # Restaurar mejores pesos
                verbose=1 # Mostrar mensajes
            ),
            ReduceLROnPlateau( # Reducir tasa de aprendizaje
                monitor='val_loss', # Reducir tasa de aprendizaje si no hay mejora
                factor=0.5, # Reducir a la mitad
                patience=10, 
                min_lr=0.00001, # Tasa mínima
                verbose=1 # Mostrar mensajes
            )
        ]
        
        # Data augmentation para entrenamiento
        datagen = ImageDataGenerator( #Aumentación de datos
            rotation_range=20,  # Rotar imágenes hasta 20 grados
            width_shift_range=0.2,  # Desplazar horizontalmente hasta 20%
            height_shift_range=0.2, # Desplazar verticalmente hasta 20%
            shear_range=0.2, # Aplicar cizallamiento
            zoom_range=0.2, # Zoom aleatorio
            horizontal_flip=True, # Voltear horizontalmente
            fill_mode='nearest' # Rellenar píxeles vacíos
        )
        x
        start_time = time.time() #
        
        try:
            # Entrenamiento desde cero
            self.history = model.fit(
                datagen.flow(X_train, y_train, batch_size=self.config.BATCH_SIZE), # batch_size 32
                steps_per_epoch=len(X_train) // self.config.BATCH_SIZE, #numero de pasos por epoca que son 32
                epochs=self.config.EPOCHS, #50 epocas
                validation_data=(X_test, y_test), #datos de validacion del archivo data_loader
                callbacks=callbacks, #Se usan los callbacks definidos arriba
                verbose=1 #mostrar el progreso del entrenamiento
            )
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")
            raise
        
        training_time = time.time() - start_time
        #Se muetra los resultado del entrenamiento
        print(f"\nEntrenamiento completado en {training_time:.2f} segundos")
        if self.history:
            print(f"Total épocas entrenadas: {len(self.history.history['loss'])}\n")
        
        return self.history
    
    #FUNCION QUE AGUARDA EL ENTRENAMIENTO 

    def save_model(self, model):        
        print("Guardando modelo...\n")
        
        # Guardar en formato .h5 (compatible con todas las versiones)
        model_path = os.path.join(self.config.MODELS_PATH, self.config.MODEL_NAME)
        
        try:
            # Método 1: Usar save_model
            save_model(model, model_path)
            print(f"Modelo guardado en: {model_path}")
        except Exception as e:
            print(f"Error usando save_model: {e}")
            # Método 2: Usar model.save() directamente
            try:
                model.save(model_path)
                print(f"Modelo guardado usando model.save(): {model_path}")
            except Exception as e2:
                print(f"Error también con model.save(): {e2}")
                return False
        
        # Intentar guardar también en formato .keras si es posible
        try:
            keras_model_path = os.path.join(self.config.MODELS_PATH, 'cat_classifier_cnn.keras')
            save_model(model, keras_model_path)
            print(f"Modelo también guardado en formato .keras: {keras_model_path}")
        except Exception as e:
            print(f"Nota: No se pudo guardar en formato .keras: {e}")
        
        # Usar el mejor modelo encontrado durante entrenamiento
        best_model_path = os.path.join(self.config.MODELS_PATH, 'best_model.h5')
        if os.path.exists(best_model_path):
            print("Usando el mejor modelo encontrado durante entrenamiento")
            try:
                # Copiar el mejor modelo sobre el actual
                import shutil
                shutil.copy2(best_model_path, model_path)
                print(f"Mejor modelo copiado a: {model_path}")
                
                # También actualizar .keras si existe
                best_keras_path = os.path.join(self.config.MODELS_PATH, 'best_model.keras')
                if os.path.exists(best_keras_path):
                    keras_model_path = os.path.join(self.config.MODELS_PATH, 'cat_classifier_cnn.keras')
                    shutil.copy2(best_keras_path, keras_model_path)
                    print(f"Mejor modelo .keras copiado")
                    
            except Exception as e:
                print(f"Error copiando mejor modelo: {e}")
        
        # Actualizar información de entrenamiento
        total_epochs = 0
        if self.history and hasattr(self.history, 'history'):
            total_epochs = len(self.history.history.get('loss', []))
        
        self.config.save_training_info(is_trained=True, epochs_completed=total_epochs)
        
        print()  # Línea en blanco
        return True
    
    def save_history(self):
        #Guarda el historial de entrenamiento
        
        if self.history is None or not hasattr(self.history, 'history'):
            print("Error: No hay historial para guardar")
            return
        
        history_path = os.path.join(self.config.RESULTS_PATH, 'training_history.pkl')
        
        try:
            with open(history_path, 'wb') as f:
                pickle.dump(self.history.history, f)
            
            print(f"Historial guardado en: {history_path}\n")
            
        except Exception as e:
            print(f"Error guardando historial: {e}\n")

    #FUNCIÓN PRINCIPAL PARA ENTRENAR 
def train_model():
    
    config = Config()
    
    try:
        # Cargar datos
        data_loader = DataLoader(config)
        data_loader.verify_data()
        X_train, X_test, y_train, y_test = data_loader.load_data()
        
        print(f"Datos cargados exitosamente")
        print(f"Entrenamiento: {len(X_train)} imágenes")
        print(f"Validación: {len(X_test)} imágenes\n")
        
    except Exception as e:
        print(f"Error cargando datos: {e}")
        raise
    
    # Crear modelo nuevo
    print("Construyendo nuevo modelo...")
    try:
        model_builder = ModelBuilder(config)
        model = model_builder.build_model()
        model_builder.compile_model(model)
        model_builder.print_model_summary(model)
    except Exception as e:
        print(f"Error construyendo modelo: {e}")
        raise
    
    # Entrenar
    trainer = ModelTrainer(config)
    try:
        trainer.train(model, X_train, X_test, y_train, y_test)
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        raise
    
    # Guardar
    trainer.save_model(model)
    trainer.save_history()
    
    print("Entrenamiento completado\n")
    
    return model, X_test, y_test