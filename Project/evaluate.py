"""
evaluate.py - Evaluación del modelo entrenado
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from config import Config

class ModelEvaluator:

    
    def __init__(self, config):
        self.config = config
    
    def evaluate_on_validation(self, model, X_test, y_test):
        #Evalúa modelo en conjunto de validación
        print("EVALUACIÓN EN VALIDACIÓN")
        
        # Predicciones
        predictions = model.predict(X_test, verbose=0)
        y_true = y_test
        y_pred = (predictions > 0.5).astype(int).flatten()
        y_pred_prob = predictions.flatten()
        
        # Métricas
        print("Reporte de Clasificación:\n")
        print(classification_report(
            y_true, y_pred,
            target_names=self.config.CLASS_NAMES.values()
        ))
        
        # AUC
        auc = roc_auc_score(y_true, y_pred_prob)
        print(f"AUC-ROC: {auc:.4f}\n")
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        print("Matriz de Confusión:")
        print(cm)
        print()
        
        return y_true, y_pred, y_pred_prob, cm
    
    def plot_confusion_matrix(self, cm):
       #Visualiza matriz de confusión
        
        print("Generando gráfica de matriz de confusión...\n")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(self.config.CLASS_NAMES.values()),
                    yticklabels=list(self.config.CLASS_NAMES.values()))
        plt.title('Matriz de Confusión')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicción')
        
        cm_path = os.path.join(self.config.RESULTS_PATH, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Matriz de confusión guardada en: {cm_path}\n")
    
    def plot_training_history(self):
    #Visualiza históricamente de entrenamiento
        
        print("Generando gráficas de entrenamiento...\n")
        
        history_path = os.path.join(self.config.RESULTS_PATH, 'training_history.pkl')
        
        if not os.path.exists(history_path):
            print("Historial no encontrado\n")
            return
        
        with open(history_path, 'rb') as f:
            history_data = pickle.load(f)
        
        # Crear figura
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Precisión
        if 'accuracy' in history_data and 'val_accuracy' in history_data:
            axes[0].plot(history_data['accuracy'], label='Entrenamiento', linewidth=2)
            axes[0].plot(history_data['val_accuracy'], label='Validación', linewidth=2)
            axes[0].set_title('Exactitud Accuracy', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Época')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Pérdida
        if 'loss' in history_data and 'val_loss' in history_data:
            axes[1].plot(history_data['loss'], label='Entrenamiento', linewidth=2)
            axes[1].plot(history_data['val_loss'], label='Validación', linewidth=2)
            axes[1].set_title('Pérdida Loss', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Época')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        results_path = os.path.join(self.config.RESULTS_PATH, 'training_results.png')
        plt.savefig(results_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráficas guardadas en: {results_path}\n")
    
    def plot_roc_curve(self, y_true, y_pred_prob):
        #Visualiza curva ROC
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        auc = roc_auc_score(y_true, y_pred_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Clasificador Aleatorio')
        plt.xlabel('Tasa de Falsos Positivos (FPR)')
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
        plt.title('Curva ROC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        roc_path = os.path.join(self.config.RESULTS_PATH, 'roc_curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Curva ROC guardada en: {roc_path}\n")

def evaluate_model(model, X_test, y_test):
    #Función principal para evaluar
    
    config = Config()
    evaluator = ModelEvaluator(config)
    
    y_true, y_pred, y_pred_prob, cm = evaluator.evaluate_on_validation(model, X_test, y_test)
    evaluator.plot_confusion_matrix(cm)
    evaluator.plot_training_history()
    evaluator.plot_roc_curve(y_true, y_pred_prob)
    
    print("Evaluación completada\n")