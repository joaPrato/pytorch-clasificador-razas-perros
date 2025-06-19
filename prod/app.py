import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import sys
import os
import pandas as pd

from utils import (
    get_breed_data,
    load_model, 
    load_breed_names, 
    preprocess_image, 
    predict_breed,
    format_breed_name,
    validate_image,
    load_model_config,
    display_breed_info,
    display_breed_sample_images,
    download_image_from_url,
    process_image_channels
)
# Agregar el directorio actual al path para importar utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Razas de Perros",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)


breed_data = get_breed_data()

# Título principal
st.title("🐶 Clasificador de Razas de Perros")
st.markdown("Sube una foto de tu perro y descubre su raza")

# Sidebar para información adicional

with st.sidebar:
    st.header("🌐 Información")
    st.markdown("""
    Esta aplicación utiliza una red neuronal entrenada 
    para identificar más de 100 razas de perros con alta precisión.
    
                
    **Instrucciones:**
    1. Sube una imagen de tu perro
    2. Espera el análisis
    3. Conoce la raza de tu perro
    """)
    
    # Separador visual
    st.divider()
    
    
    st.link_button(
        "👨🏼‍💻 Repositorio en GitHub",
        "https://github.com/joaPrato/pytorch-clasificador-razas-perros",
        help="Accede al código fuente del proyecto"
    )


# Interfaz principal
tab_upload, tab_url = st.tabs(["📁 Subir Archivo", "🌐 Desde URL"])
uploaded_image = None
image_source = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Elige una imagen de tu perro",
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Sube una imagen clara de tu perro para obtener mejores resultados"
    )

    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file)
            image_source = "archivo_subido"
            st.success("✅ Imagen cargada correctamente desde archivo")
        except Exception as e:
            st.error(f"Error al abrir la imagen: {str(e)}")
    
with tab_url:
    image_url = st.text_input(
        "Ingresa la URL de la imagen",
        placeholder="https://ejemplo.com/imagen.jpg",
        help="Pega aquí la URL de una imagen de perro." \
        "Ejemplo que peudes usar:" \
        "https://www.canal26.com/media/image/2021/07/27/481515.jpg"
    )

    if image_url:
        if st.button("🔗 Cargar imagen desde URL"):
            with st.spinner("Descargando imagen desde URL..."):
                downloaded_image = download_image_from_url(image_url)
                if downloaded_image is not None:
                    uploaded_image = downloaded_image
                    image_source = "url"
                    st.success("✅ Imagen descargada correctamente desde URL")

if uploaded_image is not None:
    # Validar la imagen
    if not validate_image(uploaded_image):
        st.error("❌ La imagen no es válida o no se puede procesar")
        st.stop()

    image = process_image_channels(uploaded_image)
    if image is None:
        st.error("❌ No se pudo procesar la imagen")
        st.stop()

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Subiste esta imagen")
        st.image(image, caption="Tu perro")
    
    with col2:
        st.subheader("Análisis")
        
        with st.spinner("Analizando imagen..."):
            # Cargar modelo y nombres de razas
            model = load_model()
            breed_names = load_breed_names()
            
            if model is not None and breed_names:
                # Preprocesar imagen
                processed_image = preprocess_image(image)

                if processed_image is not None:
                    # Realizar predicción
                    predictions = predict_breed(processed_image, model, breed_names)
                else:
                    st.error("Error en el preprocesamiento de la imagen")
            else:
                predictions = []
                st.error("Error al cargar el modelo o los nombres de razas")
        
        if predictions:
            st.success("¡Análisis completado!")
            
            # Mostrar resultados
            st.subheader("Resultados 🐾")
            
            for i, (breed, probability) in enumerate(predictions):
                formatted_breed = format_breed_name(breed)
                with st.expander(f"{i+1}. {formatted_breed} ({probability:.1%})", expanded=(i==0)):
                    display_breed_info(breed, probability,formatted_breed,breed_data)
                    st.divider()
                    display_breed_sample_images(breed, formatted_breed)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
    <p>🤖 Powered by PyTorch & ResNet-50 | 🐕 Entrenado con +10,000 imágenes</p>
    <p><em>¿Tu perro es mestizo? ¡Perfecto! Te mostraré las razas que más se le parecen</em></p>
</div>
""", unsafe_allow_html=True)   