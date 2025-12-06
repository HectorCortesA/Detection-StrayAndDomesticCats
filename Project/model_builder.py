
#Construcción del modelo CNN


from tensorflow.keras import layers, models, optimizers

class ModelBuilder:
    #Construye el modelo CNN desde cero sin transfer learning
    
    def __init__(self, config):
        self.config = config
    
    def build_model(self):
        #Crea modelo CNN personalizado desde cero
        
        print("CNN \n")
        
        model = models.Sequential([
            # Primera capa convolucional
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, self.config.IMG_CHANNELS)),
            layers.BatchNormalization(),
            #Normaliza los valores para entrenar más rápido
            #Reduce "covariate shift"
            layers.MaxPooling2D((2, 2)), #reduce tamaño de la imagen a la mitad 128x128
            layers.Dropout(0.2),
            
            # Segunda capa convolucional
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2), 
            
            # Tercera capa convolucional
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)), #reduce tamaño a 64x64
            layers.Dropout(0.3),
            
            # Cuarta capa convolucional
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)), #reduce tamaño a 32x32
            layers.Dropout(0.3), #  
            
            # Capas fully connected
            layers.Flatten(),
             #14 x14 x256 = 50, 176 valores 
            layers.Dense(256, activation='relu'),
           #Conecta los 50,176 valores → 256 neuronas
            layers.BatchNormalization(),
            #Normaliza los valores para entrenar más rápido
            #Reduce "covariate shift"
            layers.Dropout(0.4),
            
            layers.Dense(128, activation='relu'),
            #Refinamos conecta las 256 a 128 neuronas 
            layers.Dropout(0.3),
            #se apaga el 30% de la neuronas 
            
            # Capa de salida (clasificación binaria)
            #entre 1 o 0 
            layers.Dense(1, activation='sigmoid')

        ])
        return model
    
    def compile_model(self, model):
    # Se compila el modelo
        
        print("Compilando modelo\n")

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Modelo compilado exitosamente:")
        print("Optimizador: Adam lr= 0.001")
        print("Loss: binary_crossentropy")
        print("Métricas: accuracy\n")
        
        return model  # Retornar el modelo compilado
    
    def print_model_summary(self, model):
     # Imprime resumen del modelo
        
        print("Resumen del modelo:\n")
        try:
            model.summary()
        except:
            print("No se pudo mostrar el resumen, pero el modelo está construido")
        print("\n")
        
        return model  # Retornar el modelo de build_model para que pase por las capas ya con las reglas puestas de aprendizaje 