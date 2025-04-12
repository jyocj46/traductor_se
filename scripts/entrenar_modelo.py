from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Crear carpeta para guardar el modelo si no existe
os.makedirs("modelo", exist_ok=True)

# Generador de datos con validación
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Tamaño actualizado a 400x400
train = data_gen.flow_from_directory(
    "dataset", target_size=(64, 64), class_mode="categorical", subset="training"
)

val = data_gen.flow_from_directory(
    "dataset", target_size=(64, 64), class_mode="categorical", subset="validation"
)

# Red neuronal ajustada a input de 400x400
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(train.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Entrenamiento
model.fit(train, epochs=10, validation_data=val)

# Guardar modelo entrenado
model.save("modelo/modelo_entrenado.h5")
