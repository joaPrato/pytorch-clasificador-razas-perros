# 🐶 Clasificador de Razas de Perros

Una aplicación web interactiva desarrollada con **Streamlit** que utiliza una red neuronal convolucional para identificar más de 120 razas de perros a partir de fotografías.

## 📋 Características

- **Clasificación automática**: Identifica la raza de perros con alta precisión usando una red neuronal entrenada
- **Múltiples opciones de carga**: Sube imágenes desde tu dispositivo o carga desde URL
- **Información detallada**: Muestra características de cada raza (tamaño, peso, carácter, esperanza de vida)
- **Imágenes de ejemplo**: Visualiza ejemplos de cada raza identificada
- **Interfaz intuitiva**: Diseño amigable y fácil de usar
- **Análisis múltiple**: Muestra las 3 razas más probables con sus porcentajes de confianza

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **Streamlit** - Framework para aplicaciones web
- **PyTorch** - Deep Learning framework
- **torchvision** - Modelos pre-entrenados y transformaciones
- **PIL (Pillow)** - Procesamiento de imágenes
- **pandas** - Manipulación de datos
- **numpy** - Computación numérica
- **requests** - Descarga de imágenes desde URL

## 📁 Estructura del Proyecto

```
clasificador-razas-perros/
│
├── app.py                 # Aplicación principal de Streamlit
├── utils.py              # Funciones utilitarias
├── requirements.txt      # Dependencias del proyecto
├── README.md            # Este archivo
│
├── data/
│   ├── labels.csv       # Etiquetas de las razas
│   ├── breed_data.json  # Información detallada de las razas
│   └── train/           # Imágenes de entrenamiento (ejemplos)
│
└── prod/
    └── resnet50_model.pth  # Modelo entrenado
```

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.8 o superior
- Git

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/clasificador-razas-perros.git
cd clasificador-razas-perros
```

### 2. Crear Entorno Virtual (Recomendado)

```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Ejecución

### Ejecutar la Aplicación

```bash
streamlit run prod/app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Uso de la Aplicación

1. **Subir Imagen**: 
   - Usa la pestaña "📁 Subir Archivo" para cargar una imagen desde tu dispositivo
   - O usa "🌐 Desde URL" para cargar una imagen desde internet

2. **Análisis**: 
   - La aplicación procesará automáticamente la imagen
   - Mostrará las 3 razas más probables con sus porcentajes de confianza

3. **Explorar Resultados**:
   - Expande cada resultado para ver información detallada de la raza
   - Visualiza imágenes de ejemplo de cada raza
   - Usa el botón "🔄 Ver otros ejemplos" para cargar nuevas imágenes de muestra


⭐ Si este proyecto te fue útil, ¡dale una estrella en GitHub!# pytorch-clasificador-razas-perros
Red nuronal profunda para predecir la raza de un perro implementada con pytorch
