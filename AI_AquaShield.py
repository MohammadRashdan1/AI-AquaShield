import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

data = pd.read_csv("C:/Users/User/Downloads/water_leak_data1.csv")
label_encoders = {}
for column in ['Material Type', 'Maintenance History']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
X = data.drop('Leak', axis=1)
y = data['Leak']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
loss, accuracy = model.evaluate(X_test, y_test)
sample_data = np.array([[42.39,35.06,29.91,586.72, label_encoders['Material Type'].transform(['Cast Iron'])[0],12,label_encoders['Maintenance History'].transform(['Poor'])[0]]])
sample_data_scaled = scaler.transform(sample_data)
prediction = model.predict(sample_data_scaled)
for i in prediction:
    if i>=0.5:
        print("There was a leak")
    else:
        print("There was no leak")
accuracy, prediction[0][0]