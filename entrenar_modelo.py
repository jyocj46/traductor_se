import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import joblib
import tensorflow as tf

# Configura tu conexión PostgreSQL
DB_USER = "usr_traductor"
DB_PASSWORD = "oUxJEZ59sw6bPZdBBPfSvJNUBzTlkQ5f"
DB_HOST = "dpg-d0mjg0e3jp1c738dep3g-a.oregon-postgres.render.com"
DB_PORT = "5432"
DB_NAME = "lenguaje_senias_g7z9"

# Crear motor de conexión
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Leer ambas tablas
df_senas = pd.read_sql_table('datos_senas', engine)
df_numeros = pd.read_sql_table('datos_numeros', engine)

# Añadir columnas faltantes a datos_senas para tener 126 coordenadas
for i in range(21, 42):
    df_senas[f'x{i}'] = 0.0
    df_senas[f'y{i}'] = 0.0
    df_senas[f'z{i}'] = 0.0

# Unir columnas comunes
columnas = [f'{axis}{i}' for i in range(42) for axis in ('x', 'y', 'z')]
columnas.append('label')
df_total = pd.concat([df_senas[columnas], df_numeros[columnas]], ignore_index=True)

# Separar entrada (X) y salida (y)
X = df_total.drop(['label'], axis=1)
y = df_total['label']

# Codificar letras como números
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Crear modelo
model = Sequential([
    Dense(256, activation='relu', input_shape=(126,)),
    Dense(128, activation='relu'),
    Dense(len(y_categorical[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Guardar codificador
joblib.dump(encoder, "label_encoder.pkl")

# Guardar modelo 
converter = tf.lite.TFLiteConverter.from_keras_model(model)
modelo_tflite = converter.convert()
with open("modelo_senas.tflite", "wb") as f:
    f.write(modelo_tflite)

print("\n✅ Modelo entrenado y guardado como 'modelo_senas.tflite'")
print("✅ Codificador guardado como 'label_encoder.pkl'")