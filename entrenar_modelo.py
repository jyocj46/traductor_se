import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Cargar dataset
df = pd.read_csv("datos_letras.csv")

# Separar entrada (X) y salida (y)
X = df.drop('label', axis=1)
y = df['label']

# Codificar las letras (A, B, C...) como números
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Crear el modelo MLP
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dense(64, activation='relu'),
    Dense(len(y_categorical[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Guardar el modelo y el codificador
model.save("modelo_senas.h5")
import joblib
joblib.dump(encoder, "label_encoder.pkl")

print("\n✅ Modelo entrenado y guardado como 'modelo_senas.h5'")
