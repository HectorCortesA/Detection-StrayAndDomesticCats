# Clasificador de Gatos: Domésticos vs. Callejeros

Una red neuronal convolucional (CNN) para clasificar imágenes de gatos en dos categorías: **gatos domésticos**  y **gatos callejeros** .
##  Dataset

**Fuente:** [Kaggle - Pet Cats and Stray Cats 2025](https://www.kaggle.com/datasets/anonymousds2025/pet-cats-and-stray-cats-2025?resource=download)

## 📋 Tabla de Contenidos
- [Descripción](#descripción)
- [Arquitectura del Modelo](#arquitectura-del-modelo)
- [Dataset](#dataset)
- [Instalación](#instalación)
- [Uso](#uso)
- [Resultados](#resultados)
- [Tecnologías](#tecnologías)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Autores](#autores)
- [Licencia](#licencia)

## 📝 Descripción

Este proyecto implementa una CNN profunda para distinguir entre gatos domésticos y callejeros a partir de imágenes. El modelo utiliza técnicas avanzadas de aprendizaje profundo como BatchNormalization, Dropout y Data Augmentation para mejorar el rendimiento y evitar el sobreajuste.

## 🏗️ Arquitectura del Modelo

### 🔷 **Capas Convolucionales (4 capas)**

**Capa 1:**
- Filtros convolucionales (3×3) con activación ReLU
- BatchNormalization (acelera entrenamiento, reduce "covariate shift")
- MaxPooling2D (reduce dimensiones: 224px → 112px)
- Dropout (apaga neuronas aleatoriamente)

**Capas 2, 3 y 4:**
- Mismo patrón que Capa 1
- Reducciones progresivas: 112px → 56px → 28px → 14px

### 🔷 **Capa Fully Connected**

**Características:**
- Imágenes balanceadas entre ambas clases
- Preprocesadas a dimensiones uniformes (224×224px)
- División automática: 80% entrenamiento, 20% validación
