
#-=-=-=-=-=-=-=-=-=-=-#
#       IMPORT        #
#-=-=-=-=-=-=-=-=-=-=-#

import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import h5py
import cv2
from keras.models import Sequential, load_model
from PIL import Image
from glob import glob
from flask import Flask, render_template, redirect, url_for

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#  FONCTION POUR DETECTER LES VISAGES AVEC OPENCV #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

def detect_face(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# la ligne suivante est à modifier :
	face_cascade = cv2.CascadeClassifier('/home/magicmakers/.local/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
	faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors=5)
	if(len(faces) == 0):
		return None, None
	x, y, w, h = faces[0]
	return gray[y:y+w, x:x+h] #, faces[0]

#-=-=-=-=-=-=-=-=-=-=-#
#  ACTIVER LA CAMÉRA  #
#-=-=-=-=-=-=-=-=-=-=-#

cap = cv2.VideoCapture(0)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#   DICTIONNAIRE DES EMOTIONS   #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

emot = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

#-=-=-=-=-=-=-=-=-=-=-#
#   CHARGER MODÈLE    #
#-=-=-=-=-=-=-=-=-=-=-#
keras.backend.clear_session()
model = load_model("model.h5")
graph = tf.get_default_graph()

model.summary()
model._make_predict_function()

#-=-=-=-=-=-=-=-=-=-=-#
#     CRÉER APP       #
#-=-=-=-=-=-=-=-=-=-=-#

app = Flask(__name__)

#-=-=-=-=-=-=-=-=-=-=-#
#    PAGE ACCUEIL     #
#-=-=-=-=-=-=-=-=-=-=-#

@app.route('/')
def img():
	return render_template('page.html')


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
# PAGE POUR PREDIRE SUR L'IMAGE PRISE PAR LA CAMÉRA #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

@app.route('/cam/')
def cam():

	#Prend une photo avec la caméra et récupère la tête
	ret,img = cap.read()
	visage = detect_face(img)

	prediction = ""
	try:
		global graph
		with graph.as_default():
			visage = Image.fromarray(visage)
			visage = visage.convert("L")
			visage = visage.resize((48, 48))
			mon_tab = np.array(visage)
			mon_tab = mon_tab.astype("float32")
			mon_tab /= 255
			mon_tab = np.expand_dims(mon_tab,axis=0)
			mon_tab = np.expand_dims(mon_tab,axis=3)

			#Predire
			prediction = model.predict(mon_tab)
			prediction = np.argmax(prediction)
			prediction = emot[prediction]
			print(prediction)

	except Exception:
		return render_template('page.html')

	if prediction == 'anger':
		lien = "page2.html"
	if prediction == 'disgust':
		lien = "page3.html"
	if prediction == 'fear':
		lien = "page4.html"
	if prediction == 'happy':
		lien = "page5.html"
	if prediction == 'sad':
		lien = "page6.html"
	if prediction == 'surprise':
		lien = "page7.html"
	if prediction == 'neutral':
		lien = "page8.html"
	print(lien)

	return render_template(lien)
