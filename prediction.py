from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort
import os

import flask
import numpy as np
from scipy import misc
from flask import Flask, jsonify
from sklearn.externals import joblib
import pandas as pd

from keras.models import load_model
import numpy as np
from keras.preprocessing import image

from keras import backend as K

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import sys
import json
import pymysql
from urllib.request import urlopen
import time
from datetime import datetime
import pytesseract
from PIL import Image,ImageChops
import cv2
import imutils
from imutils import contours
import argparse
from flask import Flask, redirect, url_for, request, render_template,jsonify

from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort
import os
from PIL import Image,ImageChops

import flask
import glob
import cv2
import numpy as np
from scipy import misc
from flask import Flask, jsonify
from sklearn.externals import joblib
import pandas as pd

from keras.models import load_model
import numpy as np
from keras.preprocessing import image

from keras import backend as K

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import sys
import json
import pymysql
from urllib.request import urlopen
import time
from datetime import datetime
import simplejson


# Model saved with Keras model.save()
MODEL_PATH_door = '../backend_model/resources/models/door.h5'
MODEL_PATH_bumper = '../backend_model/resources/models/bumper.h5'
MODEL_PATH_hood = '../backend_model/resources/models/hood.h5'
MODEL_PATH_windshield = '../backend_model/resources/models/windshield.h5'
MODEL_PATH_doorglass = '../backend_model/resources/models/doorglass.h5'
MODEL_PATH_mirror = '../backend_model/resources/models/mirror.h5'
MODEL_PATH_roof = '../backend_model/resources/models/roof.h5'
MODEL_PATH_taillamp = '../backend_model/resources/models/taillamp.h5'
MODEL_PATH_grille = '../backend_model/resources/models/grille.h5'
MODEL_PATH_headlamp = '../backend_model/resources/models/headlamp.h5'

#Load your trained model
model_door = load_model(MODEL_PATH_door)
model_bumper = load_model(MODEL_PATH_bumper)
model_hood = load_model(MODEL_PATH_hood)
model_windshield = load_model(MODEL_PATH_windshield)
model_doorglass = load_model(MODEL_PATH_doorglass)
model_mirror = load_model(MODEL_PATH_mirror)
model_roof = load_model(MODEL_PATH_roof)
model_taillamp = load_model(MODEL_PATH_taillamp)
model_grille = load_model(MODEL_PATH_grille)
model_headlamp = load_model(MODEL_PATH_headlamp)


model_door._make_predict_function()
model_hood._make_predict_function()
model_bumper._make_predict_function()
model_windshield._make_predict_function()
model_doorglass._make_predict_function()
model_mirror._make_predict_function()
model_roof._make_predict_function()
model_taillamp._make_predict_function()
model_grille._make_predict_function()
model_headlamp._make_predict_function()


def model_predict_bumper(img_path, model):
	img = cv2.imread(img_path)
	img = cv2.resize(img,(224,224))
	img = img.astype('float32')
	img /= 255
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	preds_bumper = model_bumper.predict(x)
	return preds_bumper
	
def model_predict_door(img_path, model):
	img = cv2.imread(img_path)
	img = cv2.resize(img,(224,224))
	img = img.astype('float32')
	img /= 255
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	preds_door = model_door.predict(x)
	return preds_door

def model_predict_hood(img_path, model):
	img = cv2.imread(img_path)
	img = cv2.resize(img,(224,224))
	img = img.astype('float32')
	img /= 255
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	preds_hood = model_hood.predict(x)
	return preds_hood

def model_predict_doorglass(img_path, model):
	img = cv2.imread(img_path)
	img = cv2.resize(img,(224,224))
	img = img.astype('float32')
	img /= 255
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	preds_doorglass = model_doorglass.predict(x)
	return preds_doorglass

def model_predict_windshield(img_path, model):
	img = cv2.imread(img_path)
	img = cv2.resize(img,(224,224))
	img = img.astype('float32')
	img /= 255
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	preds_windshield = model_windshield.predict(x)
	return preds_windshield
	
def model_predict_mirror(img_path, model):
	img = cv2.imread(img_path)
	img = cv2.resize(img,(224,224))
	img = img.astype('float32')
	img /= 255
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	preds_mirror = model_mirror.predict(x)
	return preds_mirror

def model_predict_roof(img_path, model):
	img = cv2.imread(img_path)
	img = cv2.resize(img,(224,224))
	img = img.astype('float32')
	img /= 255
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	preds_roof = model_roof.predict(x)
	return preds_roof

def model_predict_grille(img_path, model):
	img = cv2.imread(img_path)
	img = cv2.resize(img,(224,224))
	img = img.astype('float32')
	img /= 255
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	preds_grille = model_grille.predict(x)
	return preds_grille

def model_predict_taillamp(img_path, model):
	img = cv2.imread(img_path)
	img = cv2.resize(img,(224,224))
	img = img.astype('float32')
	img /= 255
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	preds_taillamp = model_taillamp.predict(x)
	return preds_taillamp

def model_predict_headlamp(img_path, model):
	img = cv2.imread(img_path)
	img = cv2.resize(img,(224,224))
	img = img.astype('float32')
	img /= 255
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	preds_headlamp = model_headlamp.predict(x)
	return preds_headlamp	
	
	
def prediction(file_path,new_claim_application_id,dirName):
	
	
	# Connect with database
	dbinfo = pymysql.connect(database="cost_data", user="root", password="root", host="127.0.0.1", port=3307)

	# create row with given id
	sql = 'insert into assessment (id) values (%s) ON DUPLICATE KEY UPDATE id=%s'
	with dbinfo.cursor(pymysql.cursors.DictCursor) as cursor:
		cursor.execute(sql, (new_claim_application_id,new_claim_application_id))

	dbinfo.commit()
	dbinfo.close()

	with open(dirName+"/prediction_json.json") as f:
		prediction_json = json.load(f)
	
	# Make prediction
	preds_door = model_predict_door(file_path, model_door)
	preds_bumper = model_predict_bumper(file_path, model_bumper)
	preds_hood = model_predict_hood(file_path, model_hood)
	preds_windshield = model_predict_windshield(file_path, model_windshield)
	preds_doorglass = model_predict_doorglass(file_path, model_doorglass)
	preds_mirror = model_predict_mirror(file_path, model_mirror)
	preds_roof = model_predict_roof(file_path, model_roof)
	preds_grille = model_predict_grille(file_path, model_grille)
	preds_taillamp = model_predict_taillamp(file_path, model_taillamp)
	preds_headlamp = model_predict_headlamp(file_path, model_headlamp)

	label2index = {'Damage': 0, 'NotPresent': 1, 'Undamage': 2}
	idx2label = dict((v, k) for k, v in label2index.items())

	pred_class_door = np.argmax(preds_door)
	result_door = idx2label[pred_class_door]

	pred_class_bumper = np.argmax(preds_bumper)
	result_bumper = idx2label[pred_class_bumper]

	pred_class_hood = np.argmax(preds_hood)
	result_hood = idx2label[pred_class_hood]

	pred_class_windshield = np.argmax(preds_windshield)
	result_windshield = idx2label[pred_class_windshield]

	pred_class_doorglass = np.argmax(preds_doorglass)
	result_doorglass = idx2label[pred_class_doorglass]

	pred_class_mirror = np.argmax(preds_mirror)
	result_mirror = idx2label[pred_class_mirror]

	pred_class_roof = np.argmax(preds_roof)
	result_roof = idx2label[pred_class_roof]

	pred_class_grille = np.argmax(preds_grille)
	result_grille = idx2label[pred_class_grille]

	pred_class_taillamp = np.argmax(preds_taillamp)
	result_taillamp = idx2label[pred_class_taillamp]

	pred_class_headlamp = np.argmax(preds_headlamp)
	result_headlamp = idx2label[pred_class_headlamp]

	damaged_part = {}
	damaged_part_list = []
	confidence_list = []
	undamaged_part_list = []
	missing_part_list = []
	
	
	if pred_class_door == 0:
		confidence = preds_door[0][0]*100
		confidence = str(confidence)
		confidence = confidence.split(".")[0]
		if "door" not in prediction_json or "confidence" not in prediction_json["door"]:
			damaged_part.update({'door': 'damage'})
			prediction_json.update({'door':{'condition': 'damage','confidence':confidence}})
			damaged_part_list.append('door')
			confidence_list.append(confidence)
		elif "confidence" in prediction_json["door"]:
			pass
	elif pred_class_door == 2:
		if "door" not in prediction_json or "confidence" not in prediction_json["door"]:
			prediction_json.update({'door':{'condition':'undamage'}})
			undamaged_part_list.append('door')
		elif "confidence" in prediction_json["door"]:
			pass
	elif pred_class_door == 1:
		if "door" not in prediction_json or "confidence" not in prediction_json["door"]:
			prediction_json.update({'door':{'condition':'missing'}})
			missing_part_list.append('door')
		elif "confidence" in prediction_json["door"]:
			pass		
	else:
		pass

	if pred_class_hood == 0:
		confidence = preds_hood[0][0]*100
		confidence = str(confidence)
		confidence = confidence.split(".")[0]
		if "hood" not in prediction_json or "confidence" not in prediction_json["hood"]:
			damaged_part.update({'hood': 'damage'})
			prediction_json.update({'hood':{'condition': 'damage','confidence':confidence}})
			damaged_part_list.append('hood')
			confidence_list.append(confidence)
		elif "confidence" in prediction_json["hood"]:
			pass	
	elif pred_class_hood == 2:
		if "hood" not in prediction_json or "confidence" not in prediction_json["hood"]:
			prediction_json.update({'hood':{'condition':'undamage'}})
			undamaged_part_list.append('hood')
		elif "confidence" in prediction_json["hood"]:
			pass	
	elif pred_class_hood == 1:
		if "hood" not in prediction_json or "confidence" not in prediction_json["hood"]:
			prediction_json.update({'hood':{'condition':'missing'}})
			missing_part_list.append('hood')
		elif "confidence" in prediction_json["hood"]:
			pass		
	else:
		pass

	if pred_class_bumper == 0:
		confidence = preds_bumper[0][0]*100
		confidence = str(confidence)
		confidence = confidence.split(".")[0]
		if "bumper" not in prediction_json or "confidence" not in prediction_json["bumper"]:
			damaged_part.update({'bumper': 'damage'})
			prediction_json.update({'bumper':{'condition': 'damage','confidence':confidence}})
			damaged_part_list.append('bumper')
			confidence_list.append(confidence)
		elif "confidence" in prediction_json["bumper"]:
			pass		
	elif pred_class_bumper == 2:
		if "bumper" not in prediction_json or "confidence" not in prediction_json["bumper"]:
			prediction_json.update({'bumper':{'condition':'undamage'}})
			undamaged_part_list.append('bumper')
		elif "confidence" in prediction_json["bumper"]:
			pass	
	elif pred_class_bumper == 1:
		if "bumper" not in prediction_json or "confidence" not in prediction_json["bumper"]:
			prediction_json.update({'bumper':{'condition':'missing'}})
			missing_part_list.append('bumper')
		elif "confidence" in prediction_json["bumper"]:
			pass			
	else:
		pass

	if pred_class_windshield == 0:
		confidence = preds_windshield[0][0]*100
		confidence = str(confidence)
		confidence = confidence.split(".")[0]
		if "windshield" not in prediction_json or "confidence" not in prediction_json["windshield"]:
			damaged_part.update({'windshield': 'damage'})
			prediction_json.update({'windshield':{'condition': 'damage','confidence':confidence}})
			damaged_part_list.append('windshield')
			confidence_list.append(confidence)
		elif "confidence" in prediction_json["windshield"]:
			pass	
	elif pred_class_windshield == 2:
		
		if "windshield" not in prediction_json or "confidence" not in prediction_json["windshield"]:
			prediction_json.update({'windshield':{'condition':'undamage'}})
			undamaged_part_list.append('windshield')	
		elif "confidence" in prediction_json["windshield"]:
			pass	
	elif pred_class_windshield == 1:
		
		if "windshield" not in prediction_json or "confidence" not in prediction_json["windshield"]:
			prediction_json.update({'windshield':{'condition':'missing'}})
			missing_part_list.append('windshield')	
		elif "confidence" in prediction_json["windshield"]:
			pass			
			
	else:
		pass

	if pred_class_doorglass == 0:
		confidence = preds_doorglass[0][0]*100
		confidence = str(confidence)
		confidence = confidence.split(".")[0]
		if "doorglass" not in prediction_json or "confidence" not in prediction_json["doorglass"]:
			damaged_part.update({'doorglass': 'damage'})
			prediction_json.update({'doorglass':{'condition': 'damage','confidence':confidence}})
			damaged_part_list.append('doorglass')
			confidence_list.append(confidence)
		elif "confidence" in prediction_json["doorglass"]:
			pass	
	elif pred_class_doorglass == 2:
		
		if "doorglass" not in prediction_json or "confidence" not in prediction_json["doorglass"]:
			prediction_json.update({'doorglass':{'condition':'undamage'}})
			undamaged_part_list.append('doorglass')	
		elif "confidence" in prediction_json["doorglass"]:
			pass	
	elif pred_class_doorglass == 1:
		
		if "doorglass" not in prediction_json or "confidence" not in prediction_json["doorglass"]:
			prediction_json.update({'doorglass':{'condition':'missing'}})
			missing_part_list.append('doorglass')	
		elif "confidence" in prediction_json["doorglass"]:
			pass		
	else:
		pass

	if pred_class_mirror == 0:
		confidence = preds_mirror[0][0]*100
		confidence = str(confidence)
		confidence = confidence.split(".")[0]
		if "mirror" not in prediction_json or "confidence" not in prediction_json["mirror"]:
			damaged_part.update({'mirror': 'damage'})
			prediction_json.update({'mirror':{'condition': 'damage','confidence':confidence}})
			damaged_part_list.append('mirror')
			confidence_list.append(confidence)
		elif "confidence" in prediction_json["mirror"]:
			pass	
	elif pred_class_mirror == 2:
		
		if "mirror" not in prediction_json or "confidence" not in prediction_json["mirror"]:
			prediction_json.update({'mirror':{'condition':'undamage'}})
			undamaged_part_list.append('mirror')	
		elif "confidence" in prediction_json["mirror"]:
			pass		
	elif pred_class_mirror == 1:
		
		if "mirror" not in prediction_json or "confidence" not in prediction_json["mirror"]:
			prediction_json.update({'mirror':{'condition':'missing'}})
			missing_part_list.append('mirror')	
		elif "confidence" in prediction_json["mirror"]:
			pass			
	else:
		pass

	if pred_class_roof == 0:
		confidence = preds_roof[0][0]*100
		confidence = str(confidence)
		confidence = confidence.split(".")[0]
		if "roof" not in prediction_json or "confidence" not in prediction_json["roof"]:
			damaged_part.update({'roof': 'damage'})
			prediction_json.update({'roof':{'condition': 'damage','confidence':confidence}})
			damaged_part_list.append('roof')
			confidence_list.append(confidence)
		elif "confidence" in prediction_json["roof"]:
			pass	
	elif pred_class_roof == 2:
		
		if "roof" not in prediction_json or "confidence" not in prediction_json["roof"]:
			prediction_json.update({'roof':{'condition':'undamage'}})
			undamaged_part_list.append('roof')	
		elif "confidence" in prediction_json["roof"]:
			pass
	elif pred_class_roof == 1:
		
		if "roof" not in prediction_json or "confidence" not in prediction_json["roof"]:
			prediction_json.update({'roof':{'condition':'missing'}})
			missing_part_list.append('roof')	
		elif "confidence" in prediction_json["roof"]:
			pass		
	else:
		pass

	if pred_class_grille == 0:
		confidence = preds_grille[0][0]*100
		confidence = str(confidence)
		confidence = confidence.split(".")[0]
		if "grille" not in prediction_json or "confidence" not in prediction_json["grille"]:
			damaged_part.update({'grille': 'damage'})
			prediction_json.update({'grille':{'condition': 'damage','confidence':confidence}})
			damaged_part_list.append('grille')
			confidence_list.append(confidence)
		elif "confidence" in prediction_json["grille"]:
			pass	
	elif pred_class_grille == 2:
		
		if "grille" not in prediction_json or "confidence" not in prediction_json["grille"]:
			prediction_json.update({'grille':{'condition':'undamage'}})
			undamaged_part_list.append('grille')
		elif "confidence" in prediction_json["grille"]:
			pass
	elif pred_class_grille == 1:
		
		if "grille" not in prediction_json or "confidence" not in prediction_json["grille"]:
			prediction_json.update({'grille':{'condition':'missing'}})
			missing_part_list.append('grille')
		elif "confidence" in prediction_json["grille"]:
			pass		
	else:
		pass

	if pred_class_headlamp == 0:
		confidence = preds_headlamp[0][0]*100
		confidence = str(confidence)
		confidence = confidence.split(".")[0]
		if "headlamp" not in prediction_json or "confidence" not in prediction_json["headlamp"]:
			damaged_part.update({'headlamp': 'damage'})
			prediction_json.update({'headlamp':{'condition': 'damage','confidence':confidence}})
			damaged_part_list.append('headlamp')
			confidence_list.append(confidence)
		elif "confidence" in prediction_json["headlamp"]:
			pass	
	elif pred_class_headlamp == 2:
		
		if "headlamp" not in prediction_json or "confidence" not in prediction_json["headlamp"]:
			prediction_json.update({'headlamp':{'condition':'undamage'}})
			undamaged_part_list.append('headlamp')	
		elif "confidence" in prediction_json["headlamp"]:
			pass	
	elif pred_class_headlamp == 1:
		
		if "headlamp" not in prediction_json or "confidence" not in prediction_json["headlamp"]:
			prediction_json.update({'headlamp':{'condition':'missing'}})
			missing_part_list.append('headlamp')	
		elif "confidence" in prediction_json["headlamp"]:
			pass			
	else:
		pass

	if pred_class_taillamp == 0:
		confidence = preds_taillamp[0][0]*100
		confidence = str(confidence)
		confidence = confidence.split(".")[0]
		if "taillamp" not in prediction_json or "confidence" not in prediction_json["taillamp"]:
			damaged_part.update({'taillamp': 'damage'})
			prediction_json.update({'taillamp':{'condition': 'damage','confidence':confidence}})
			damaged_part_list.append('taillamp')
			confidence_list.append(confidence)
		elif "confidence" in prediction_json["taillamp"]:
			pass	
	elif pred_class_taillamp == 2:
		
		if "taillamp" not in prediction_json or "confidence" not in prediction_json["taillamp"]:
			prediction_json.update({'taillamp':{'condition':'undamage'}})
			undamaged_part_list.append('taillamp')	
		elif "confidence" in prediction_json["taillamp"]:
			pass
	elif pred_class_taillamp == 1:
		
		if "taillamp" not in prediction_json or "confidence" not in prediction_json["taillamp"]:
			prediction_json.update({'taillamp':{'condition':'missing'}})
			missing_part_list.append('taillamp')	
		elif "confidence" in prediction_json["taillamp"]:
			pass		
	else:
		pass

		
	#f = open(dirName+'/damaged_part_list.txt','w')
	#simplejson.dump(damaged_part_list,f)
	#f.close()
	
	#f = open(dirName+'/confidence_list.txt','w')
	#simplejson.dump(confidence_list,f)
	#f.close()
	
	#f = open(dirName+'/undamaged_part_list.txt','w')
	#simplejson.dump(undamaged_part_list,f)
	#f.close()
		
	with open(dirName+'/prediction_json.json', 'w') as prediction_output:
		json.dump(prediction_json, prediction_output)

	with open('json_data/damaged_parts.json', 'w') as damage:
		json.dump(damaged_part, damage)

	response = json.dumps(prediction_json)
	return response