
#-=-=-=-=-=-=-=-=-=-=-#
#       IMPORT        #
#-=-=-=-=-=-=-=-=-=-=-#

import csv
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import numpy as np
import h5py
from PIL import Image

#-=-=-=-=-=-=-=-=-=-=-#
#        DATA         #
#-=-=-=-=-=-=-=-=-=-=-#

mes_labels = []
mon_tab = []
X_train = []
y_train = []
x_test = []
y_test = []
x_train = mon_tab
y_train = mes_labels

# Ouverture des tableaux de visages

with open('../Data/train.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		#print(row['Pixels'])
		mon_image = row['Pixels'].split(" ")
		mon_image = np.array(mon_image)
		mon_image = mon_image.reshape((48,48))
		mon_image = mon_image.astype("float32")
		mon_image /= 255
		#print(mon_image.shape)
		mon_tab.append(mon_image)

mon_tab = np.array(mon_tab)

# Ouverture du tableau d'émotions

with open('../Data/train.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		emotion = row['Emotion']
		mes_labels.append(emotion)

		#anger=0
		#disgust=1
		#fear=2
		#happy=3
		#sad=4
		#surprise=5
		#neutral=6

mes_labels = np.array(mes_labels)
mes_labels = mes_labels.astype('uint8')

print(mes_labels)
print(mon_tab)

mon_tab, mes_labels = shuffle(mon_tab, mes_labels)
x_train, x_test, y_train, y_test = train_test_split(mon_tab, mes_labels, test_size = 0.2)

x_train = np.array(x_train)
x_test= np.array(x_test)

print(x_train.shape)

x_train = x_train.reshape(3342,48, 48, 1)
x_test = x_test.reshape(836,48, 48, 1)

y_train= to_categorical(y_train,7)
y_test= to_categorical(y_test,7)


#-=-=-=-=-=-=-=-=-=-=-#
#       MODÈLE        #
#-=-=-=-=-=-=-=-=-=-=-#

model = Sequential()
model.add(Conv2D(24, kernel_size=9, strides=2, input_shape=(48, 48, 1), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(32, kernel_size=9, strides=2, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(64, kernel_size=9, strides=2, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())
model.add(Dense(700, activation="relu"))
model.add(Dense(7, activation="softmax"))

model.compile(loss="categorical_crossentropy",
				optimizer=Adam(),
				metrics=["accuracy"])

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#       TRAIN - ENTRAÎNEMENT        #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

model.fit(x_train, y_train,
		batch_size = 510,
		epochs = 100,
		validation_data = (x_test, y_test))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#       SAUVEGARDER MODÈLE        #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

model.save("model.h5")
