import tensorflow as tf

# Cargar el modelo Keras entrenado
modelo = tf.keras.models.load_model("modelo_senas.h5")

# Convertir a TensorFlow Lite
conversor = tf.lite.TFLiteConverter.from_keras_model(modelo)
modelo_tflite = conversor.convert()

# Guardar el modelo convertido
with open("modelo_senas.tflite", "wb") as f:
    f.write(modelo_tflite)

print("✅ Modelo convertido y guardado como 'modelo_senas.tflite'")
