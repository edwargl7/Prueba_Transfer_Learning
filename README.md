# Prueba_Transfer_Learning
Prueba de Transfer Learning basado en el tutorial de Tensorflow.

En este tutorial se usa Tensorflow Hub, para usar el modelo entrenado de mobilenet. El script transfer_learning_simple.py cuenta con el código del tutorial de tensorflow.

Pasos para ejecutarlo:
- Para ejecutar con GPU realizar la instalación de CUDA, sigue este enlace:https://www.tensorflow.org/install/gpu
- Clonar el repositorio, y ubicarse dentro del mismo.
- Crear el ambiente virtual usando virtualenv.
  ```sh
   virtualenv -p python3 env
  ```
- Instalar las librerías necesarias incluidas en el archivo requirements.txt.
  ```sh
  pip3 install -r requirements.txt
  ```
- Ejecute el script.
  ```sh
   python3 transfer_learning_simple.py
  ```
con imágenes de gaseosas.

# Referencia
- https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub
