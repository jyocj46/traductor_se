import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import joblib

# Configura tu conexión PostgreSQL
DB_USER = "usr_traductor"
DB_PASSWORD = "NVILfbkCxpEqMNWLAJBY9JxnDSSaUNXy"
DB_HOST = "dpg-d044scruibrs73aoldfg-a.oregon-postgres.render.com"
DB_PORT = "5432"
DB_NAME = "lenguaje_senias"

# Crear motor de conexión
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Leer datos desde la tabla
df = pd.read_sql_table('datos_senas', engine)

# Separar entrada (X) y salida (y)
X = df.drop(['id', 'label'], axis=1)
y = df['label']

# Codificar letras como números
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Crear modelo
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dense(64, activation='relu'),
    Dense(len(y_categorical[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Guardar modelo y codificador
model.save("modelo_senas.h5")
joblib.dump(encoder, "label_encoder.pkl")

print("\n✅ Modelo entrenado y guardado como 'modelo_senas.h5'")