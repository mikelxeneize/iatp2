#importacion de las bibliotecas de Keras y librerias de Keras

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importacion del set de entrenamiento

dataset_train_sinFiltrar = pd.read_csv('Weather_Valencia_Train_2015-2017.csv')

#filtracion de las columnas que me interesan
dataset_train =dataset_train_sinFiltrar.iloc[:, 2:3]
print(dataset_train)
training_set = dataset_train.values

# Escalado de caracteristicas

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) # normalizacion de los datos
training_set_scaled = sc.fit_transform(training_set)

#Creación de una estructura de datos con 120 (equivale a 5 dias previos) pasos de tiempo y 1 salida

X_train = []
y_train = []
for i in range(120, 26366): #lee a partir  de la fila 120 hasta la cantidad de filas del dataset (26366)
    X_train.append(training_set_scaled[i-120:i, 0])  #para mi aca no va 0 
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train) #los transforma en np.array

#Remodelación

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) ##aca para meterle mas parametros al tensor

#Importación de las librerías y paquetes Keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Inicializando el RNN

regressor = Sequential()

#Añadiendo la primera capa de LSTM y algo de regularización de la deserción

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Añadiendo la segunda capa de LSTM y algo de regularización de la deserción

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Añadiendo la tercera capa de LSTM y algo de regularización de la deserción

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Añadiendo la cuarta capa de LSTM y algo de regularización de la deserción

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Añadiendo la capa de salida

regressor.add(Dense(units = 1))

#Compilación del RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Ajustar el RNN al set de entrenamiento

regressor.fit(X_train, y_train, epochs = 1, batch_size = 32)

#Obtención del clima
horas_a_predecir = 480 #cantidad de horas a predecir
dataset_test = pd.read_csv('Weather_Valencia_Test_2018.csv')
real_weather_temp = dataset_test.iloc[horas_a_predecir, 2:3].values

#Obtención del precio de las acciones previsto para 2017

dataset_total = pd.concat((dataset_train['temp'], dataset_test['temp']), axis = 0) #concatena los datos de entrenamiento y los de test
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 120:].values #toma los ultimos 120 HORAS del 2016
inputs = inputs.reshape(-1,1) #transforma los datos en un array de 1 columna
inputs = sc.transform(inputs) #normaliza los datos
X_test = [] #crea un array vacio
for i in range(120, horas_a_predecir): #toma los ultimos 120 dias del 2016 8780
    X_test.append(inputs[i-120:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_weather_temp = regressor.predict(X_test)
predicted_weather_temp = sc.inverse_transform(predicted_weather_temp)

#Visualización de los resultados

xgrafico = np.arange(0, horas_a_predecir, 1)
print(xgrafico.shape)
print(real_weather_temp.shape)
print(predicted_weather_temp.shape)
plt.scatter(xgrafico,real_weather_temp, color = 'red', label = 'Real weather temperatura')
plt.scatter(xgrafico,predicted_weather_temp, color = 'blue', label = 'Predicted weather temperatura')
plt.title('Real weather temperatura')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()