import streamlit as st
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import io
import sys
import os
import pandas as pd
import requests
from urllib.parse import urlparse
import tempfile


def load_model_weights(model_path, model_architecture):
    """
    Cargar los pesos del modelo entrenado
    
    Args:
        model_path (str): Ruta al archivo del modelo
        model_architecture: Arquitectura del modelo
    
    Returns:
        model: Modelo cargado con los pesos
    """
    try:
        model_architecture.load_state_dict(torch.load(model_path, map_location='cpu'))
        model_architecture.eval()
        return model_architecture
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

def preprocess_image_pytorch(image, image_size=(224, 224)):
    """
    Preprocesar imagen para modelos PyTorch
    
    Args:
        image (PIL.Image): Imagen PIL
        image_size (tuple): Tamaño objetivo (height, width)
    
    Returns:
        torch.Tensor: Tensor preprocesado
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

def get_class_names():
    """
    Obtener los nombres de las clases/razas
    Modifica esta función según tus clases específicas
    
    Returns:
        list: Lista de nombres de razas
    """
    # Reemplaza con las razas que tu modelo puede clasificar
    return [
        "Golden Retriever",
        "Labrador Retriever", 
        "Pastor Alemán",
        "Bulldog Francés",
        "Beagle",
        # Agrega aquí todas las razas que tu modelo puede clasificar
    ]

def postprocess_predictions(outputs, class_names, top_k=3):
    """
    Postprocesar las salidas del modelo para obtener las top-k predicciones
    
    Args:
        outputs (torch.Tensor): Salidas del modelo
        class_names (list): Lista de nombres de clases
        top_k (int): Número de predicciones top a retornar
    
    Returns:
        list: Lista de tuplas (clase, probabilidad)
    """
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    probabilities = probabilities.cpu().numpy()[0]
    
    # Obtener los índices de las top-k predicciones
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    
    # Crear lista de tuplas (clase, probabilidad)
    top_predictions = [(class_names[i], probabilities[i]) for i in top_indices]
    
    return top_predictions

def validate_image(image):
    # Verificar que la imagen tenga el formato correcto
    if image.mode != 'RGB':
        try:
            image = image.convert('RGB')
        except:
            return False
    
    # Verificar dimensiones mínimas
    width, height = image.size
    if width < 32 or height < 32:
        return False
        
    return True

# Función de ejemplo para cargar configuración del modelo
def load_model_config(config_path=None):
    """
    Cargar configuración del modelo
    
    Args:
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        dict: Diccionario con la configuración
    """
    # Configuración por defecto
    default_config = {
        'input_size': (224, 224),
        'num_classes': len(get_class_names()),
        'model_type': 'resnet50',
        'dropout': 0.5
    }
    
    if config_path and os.path.exists(config_path):
        # Aquí podrías cargar desde un archivo JSON o YAML
        # import json
        # with open(config_path, 'r') as f:
        #     config = json.load(f)
        # return config
        pass
    
    return default_config

def load_breed_names():
    """
    Cargar los nombres de las razas desde el archivo labels.csv
    manteniendo el mismo orden que se usó durante el entrenamiento
    """
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        labels_path = os.path.join(BASE_DIR, '..', 'data', 'labels.csv')
        labels_df = pd.read_csv(labels_path)
        breeds = sorted(labels_df['breed'].unique())
        return breeds
        
    except Exception as e:
        st.error(f"Error al cargar nombres de razas: {str(e)}")
        return []

@st.cache_resource
def load_model():
    """Cargar el modelo entrenado de PyTorch"""
    try:
        # Cargar el modelo desde la carpeta prod/
        model_path = os.path.join('prod', 'resnet50_model.pth')
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
                param.requires_grad = False
        num_features = model.fc.in_features
        num_classes=120
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5), 
            torch.nn.Linear(model.fc.in_features, num_classes)
        )
        
        # Cargar los pesos entrenados
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            return model
        else:
            st.error(f"No se encontró el modelo en {model_path}")
            return None
            
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocesar la imagen para el modelo PyTorch"""
    # Transformaciones estándar para modelos pre-entrenados
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Convertir imagen PIL a tensor
    image_tensor = transform(image).unsqueeze(0)  # Añadir dimensión batch
    
    return image_tensor

def predict_breed(image_tensor, model, breed_names, min_probability=0.10, max_results=3):
    try:
        if image_tensor is None:
            return []
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()[0]
        
        # Obtener todos los índices ordenados por probabilidad (de mayor a menor)
        sorted_indices = np.argsort(probabilities)[::-1]
        
        # Filtrar predicciones que superen el umbral mínimo
        filtered_breeds = []
        for idx in sorted_indices:
            probability = probabilities[idx]
            
            # Solo incluir si supera el umbral mínimo
            if probability >= min_probability:
                breed_name = breed_names[idx] if idx < len(breed_names) else f"Raza_{idx}"
                filtered_breeds.append((breed_name, probability))
                
                # Limitar al número máximo de resultados
                if len(filtered_breeds) >= max_results:
                    break
            else:
                break
        
        # Si no hay predicciones que superen el umbral, devolver la mejor predicción
        if not filtered_breeds:
            best_idx = sorted_indices[0]
            best_breed = breed_names[best_idx] if best_idx < len(breed_names) else f"Raza_{best_idx}"
            best_probability = probabilities[best_idx]
            filtered_breeds = [(best_breed, best_probability)]
            
            # Mostrar advertencia sobre baja confianza
            st.warning(f"⚠️ Predicción con baja confianza ({best_probability:.1%}). ")
        return filtered_breeds
        
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        return []

def format_breed_name(breed_name):
    """Formatear el nombre de la raza para mostrar"""
    # Reemplazar guiones bajos con espacios y capitalizar
    formatted_name = breed_name.replace('_', ' ').title()
    return formatted_name

def load_breed_data(json_file_path):
    """Cargar datos de razas desde archivo JSON"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo JSON: {json_file_path}")
        return {}
    except json.JSONDecodeError:
        st.error("Error al leer el archivo JSON. Verifica el formato.")
        return {}
    
@st.cache_data
def get_breed_data():
    """Cargar y cachear los datos de razas"""
    return load_breed_data(os.path.join('data', 'breed_data.json'))

@st.cache_data
def load_labels_dataframe():
    """Cargar y cachear el DataFrame de labels.csv"""
    try:
        labels_path = os.path.join('data', 'labels.csv')
        return pd.read_csv(labels_path)
    except Exception as e:
        st.error(f"Error al cargar labels.csv: {str(e)}")
        return pd.DataFrame()

def get_breed_sample_images(breed_name, max_images=4):
    """
    Obtener imágenes de ejemplo de una raza específica
    
    Args:
        breed_name (str): Nombre de la raza
        max_images (int): Número máximo de imágenes a mostrar
    
    Returns:
        list: Lista de rutas a las imágenes de ejemplo
    """
    try:
        # Cargar el DataFrame de labels
        labels_df = load_labels_dataframe()
        
        if labels_df.empty:  
            return []
        
        # Filtrar imágenes de la raza específica
        breed_images = labels_df[labels_df['breed'] == breed_name]
        
        if breed_images.empty:
            return []
        
        # Obtener una muestra aleatoria de imágenes
        sample_size = min(max_images, len(breed_images))
        sampled_images = breed_images.sample(n=sample_size)
        
        # Construir rutas completas a las imágenes
        train_dir = os.path.join('data', 'train')
        
        image_paths = []
        for _, row in sampled_images.iterrows():
            image_id = row['id']
            
            # Intentar con diferentes extensiones de archivo
            possible_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            
            found_image = False
            for ext in possible_extensions:
                image_path = os.path.join(train_dir, image_id + ext)
                
                # Verificar que el archivo existe
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    found_image = True
                    break
            
            # Si no se encuentra con extensiones, intentar sin extensión (por si acaso)
            if not found_image:
                image_path = os.path.join(train_dir, image_id)
                if os.path.exists(image_path):
                    image_paths.append(image_path)
        
        return image_paths
        
    except Exception as e:
        st.error(f"Error al obtener imágenes de ejemplo: {str(e)}")
        return []

def display_breed_sample_images(breed_name, formatted_breed_name):
    """
    Mostrar imágenes de ejemplo de la raza
    
    Args:
        breed_name (str): Nombre interno de la raza
        formatted_breed_name (str): Nombre formateado de la raza
    """
    #st.subheader(f"📸 Ejemplos de {formatted_breed_name}")
    
    # Obtener imágenes de ejemplo
    sample_images = get_breed_sample_images(breed_name, max_images=4)
    
    if not sample_images:
        st.info(f"No se encontraron imágenes de ejemplo para la raza '{breed_name}'.")
        return
    
    # Mostrar las imágenes en columnas
    cols = st.columns(min(len(sample_images), 4))
    
    for i, image_path in enumerate(sample_images):
        try:
            image = Image.open(image_path)
            with cols[i % 4]:
                st.image(
                    image, 
                    caption=f"Ejemplo {i+1}",
                    
                )
        except Exception as e:
            st.error(f"Error al cargar imagen {image_path}: {str(e)}")

    # Botón para cargar nuevas imágenes (único para cada raza)
    button_key = f"reload_images_{breed_name}_{hash(formatted_breed_name) % 10000}"
    if st.button(f"🔄 Mostrar otros ejemplos", key=button_key):
        st.rerun()


def display_breed_info(breed_name, probability, formatted_breed, breed_data):
    """ Mostrar información de la raza desde JSON """

    # Buscar información de la raza
    info = breed_data.get(breed_name, None)

    # Información por defecto si no se encuentra la raza
    if info is None:
        info = {
            "tamaño": "No disponible",
            "peso": "No disponible", 
            "altura": "No disponible",
            "carácter": "No disponible",
            "esperanza_vida": "No disponible",
            "descripcion": "Información no disponible para esta raza."
        }
        st.warning(f"⚠️ No se encontró información específica para {formatted_breed}")
    
    # Mostrar la información
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Probabilidad", f"{probability:.1%}")
    
    with col2:
        st.subheader(f"🐕 {formatted_breed}")
    
    if info:
        st.write(f"**Descripción:** {info['descripcion']}")
        
        # Crear columnas para las características
        char_col1, char_col2, char_col3 = st.columns(3)
        
        with char_col1:
            st.write(f"**Tamaño:** {info['tamaño']}")
            st.write(f"**Peso:** {info['peso']}")
        
        with char_col2:
            st.write(f"**Altura:** {info['altura']}")
            st.write(f"**Esperanza vida:** {info['esperanza_vida']}")
        
        with char_col3:
            st.write(f"**Carácter:** {info['carácter']}")


def download_image_from_url(url, timeout=10):
    try:
        # Validar que la URL tenga un formato válido
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            st.error("URL no válida. Asegúrate de incluir http:// o https://")
            return None
        
        # Realizar la petición
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Verificar que el contenido sea una imagen
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            st.error("La URL no parece contener una imagen válida")
            return None
        
        # Cargar imagen desde bytes
        image_bytes = io.BytesIO(response.content)
        image = Image.open(image_bytes)
        
        return image
        
    except requests.exceptions.Timeout:
        st.error("Tiempo de espera agotado. Intenta con otra URL.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error al descargar la imagen: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error al procesar la imagen desde URL: {str(e)}")
        return None

def process_image_channels(image):
    """
    Procesar imagen para asegurar que tenga 3 canales RGB
    
    Args:
        image (PIL.Image): Imagen original
    
    Returns:
        PIL.Image: Imagen procesada en formato RGB
    """
    try:
        # Si la imagen tiene 4 canales (RGBA), convertir a RGB
        if image.mode == 'RGBA':
            # Crear un fondo blanco
            background = Image.new('RGB', image.size, (255, 255, 255))
            # Pegar la imagen RGBA sobre el fondo blanco usando el canal alpha
            background.paste(image, mask=image.split()[-1])  # Usar canal alpha como máscara
            image = background
        
        # Si la imagen está en escala de grises, convertir a RGB
        elif image.mode == 'L':
            image = image.convert('RGB')
        
        # Si la imagen tiene modo P (palette), convertir a RGB
        elif image.mode == 'P':
            image = image.convert('RGB')
        
        # Para otros modos, intentar convertir a RGB
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        st.error(f"Error al procesar canales de imagen: {str(e)}")
        return None