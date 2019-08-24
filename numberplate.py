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
import DetectChars
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
import DetectPlates
import PossiblePlate

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







def numberplate123(path,file_path,new_claim_application_id):
	blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()  # attempt KNN training
	
	
	
	
	if blnKNNTrainingSuccessful == False:  # if KNN training was not successful
		print("\nerror: KNN traning was not successful\n")  # show error message
		return  # and exit program
	# end if
	for filename in os.listdir(path):
		print(filename)
		imgOriginalScene = cv2.imread(file_path)  # open image

		if imgOriginalScene is None:  # if image was not read successfully
			# return("\nerror: image not read from file \n\n") # print error message to std out
			return ("image not read from file")
			os.system("pause")  # pause so user can see error message
			continue  # and exit program
		# end if

		listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)  # detect plates

		listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)  # detect chars in plates

		# cv2.imshow("imgOriginalScene", imgOriginalScene)            # show scene image

		if len(listOfPossiblePlates) == 0:  # if no plates were found
			return ("no license plates were detected")
		# return("\nno license plates were detected\n")  # inform user no plates were found
		# path = 'output5'
		# cv2.imwrite(os.path.join(path, filename), imgOriginalScene)
		else:  # else
			# if we get in here list of possible plates has at leat one plate

			# sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
			listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

			# suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
			licPlate = listOfPossiblePlates[0]
			path1 = path+"/output5"
			if os.path.exists(path1):
				delete_previous_images = os.listdir(path1)
				for item in delete_previous_images:
					os.remove(os.path.join(path1, item))
			else:
				os.makedirs(path1)
			# cv2.imshow("imgPlate", licPlate.imgPlate)           # show crop of plate and threshold of plate
			#path1 = path+'/output5'
			filename=new_claim_application_id+"_"+filename
			print("filename= ",filename)
			cv2.imwrite(os.path.join(path1, filename), licPlate.imgPlate)
			# return send_file(filename, mimetype='image/gif')
			# cv2.imshow("imgThresh", licPlate.imgThresh)
			# path = 'output_thresh1/'
			# cv2.imwrite(os.path.join(path, "one_"+filename), licPlate.imgThresh)

			if len(licPlate.strChars) == 0:  # if no chars were found in the plate
				return ("no characters were detected")
				# return("\nno characters were detected\n\n")  # show message
				continue  # and exit program
			# end if

			drawRedRectangleAroundPlate(imgOriginalScene, licPlate)  # draw red rectangle around plate
			result = "license plate read from image"

			# return("\nlicense plate read from image = " + licPlate.strChars + "\n")  # write license plate text to std out
			# print("----------------------------------------")

			# writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # write license plate text on the image

			# cv2.imshow("imgOriginalScene", imgOriginalScene)                # re-show scene image

			cv2.imwrite(path+"/imgOriginalScene.png", imgOriginalScene)  # write image out to file
			
			return_number_plate= numberplate(path1,new_claim_application_id)
			return return_number_plate

		# end if else

		# cv2.waitKey(0)					# hold windows open until user presses a key


	# end main


def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
	SCALAR_BLACK = (0.0, 0.0, 0.0)
	SCALAR_WHITE = (255.0, 255.0, 255.0)
	SCALAR_YELLOW = (0.0, 255.0, 255.0)
	SCALAR_GREEN = (0.0, 255.0, 0.0)
	SCALAR_RED = (0.0, 0.0, 255.0)

	p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)  # get 4 vertices of rotated rect

	cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)  # draw 4 red lines
	cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
	cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
	cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)


# end function
avgconf_text = {}
confmax = []
textmax = []
average_conf = []

#apply OCR on number plate
def numberplate(path1,new_claim_application_id):
	
	#dirName = path+"/output5"
	#if os.path.exists(dirName):
		#pass
	#else:
		#os.makedirs(dirName)
	
	for filename in os.listdir(path1):
		print(filename)
		dummy_uid=filename.split("_")[0]
		print(dummy_uid)
		if dummy_uid==new_claim_application_id:
			im = Image.open(path1+"/" + filename)
			def trim(im):
				bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
				diff = ImageChops.difference(im, bg)
				diff = ImageChops.add(diff, diff, 2.0, -100)
				bbox = diff.getbbox()
				if bbox:
					return im.crop(bbox)

			# trim(im).show()
			y11 = trim(im).save(path1+"/crop.png")
			image = cv2.imread(path1+"/crop.png")

			
			
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			gray1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
			filename = "new_{}".format(filename)
			img_process_numberplate(filename, image,path1)
			#print("BINARY IMAGE")
			#img_process(filename,gray1)
			gray2 = cv2.medianBlur(gray, 3)
			#print("GRAY IMAGE")
			img_process_numberplate(filename,gray2,path1)
			gray3 = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)[1]
			#print("BINARY INVERSION IMAGE")
			img_process_numberplate(filename, gray3,path1)
			#gray4 = cv2.threshold(gray2,127,255,cv2.THRESH_BINARY)[1]
			#print("GLOBAL THRESHOLDED IMAGE")
			#img_process(filename, gray4)
			gray5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)[1]
			#print("THRESH TRUNCATED IMAGE")
			img_process_numberplate(filename, gray5,path1)
			#6
			gray6 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)[1]
			#print("THRESH TOZERO IMAGE")
			img_process_numberplate(filename, gray6,path1)
			#7
			#gray7 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)[1]
			#print("THRESH TOZERO INVERTED IMAGE")
			#img_process(filename, gray7)
			#8
			#gray8 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
			#                            cv2.THRESH_BINARY, 11, 2)
			#print("ADAPTIVE MEAN THRESHOLDING IMAGE")
			#img_process(filename, gray8)
			#9
			#gray9 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
			#                           cv2.THRESH_BINARY, 11, 2)
			#print("THRESH TRUNCATED IMAGE")
			#img_process(filename, gray9)
			#10
			#gray10 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
			#print("THRESH OTSU IMAGE")
			#img_process(filename, gray10)
			#11
			blur = cv2.GaussianBlur(gray, (5, 5), 0)
			gray11 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
			#print("THRESH OTSU BLUR IMAGE")
			img_process_numberplate(filename, gray11,path1)
			#print("confidence :", confmax, "text is:", textmax)
			#print(average_conf)
			max_avg_conf = max(average_conf)
			ind = average_conf.index(max_avg_conf)
			#print(ind)
			pre_txt = textmax[ind]
			#print("Avg confidence : ",max_avg_conf," text ",pre_txt)
			#print(pre_txt)
			for i in range(len(average_conf)):
				avgconf_text[average_conf[i]]=textmax[i]
			sorted_text=[]
			#print(avgconf_text)
			x1=sorted(avgconf_text.keys(), reverse=True)
			j=0
			for i in x1:
				if j<3:
					sorted_text.append(avgconf_text[i])
				j=j+1

			#print(sorted_text)
			j=0

			li=[]
			m=0

			for i in sorted_text:

				s=''
				if i is not '' and i is not ' ' and i is not '  ':
					s=" ".join(str(j) for j in i)

					al_num=''.join(e for e in s if e.isalnum())
					c=''.join(x for x in al_num if not x.islower())

					data = {}
					for i in range(0,len(c)):

						if c[i].isalpha():
							data[m] = c[i:]
							m=m+1
							li.append(data)
							print(c[i:],end="\n")
							break
			
			for i in textmax:
				print(i)
				
			avgconf_text.clear()
			del confmax[:]
			del textmax[:]
			del average_conf[:]	
			
			return jsonify(li)
		else:
			return "error"


def img_process_numberplate(filename, gray1, path1):
	cv2.imwrite(path1+'/'+filename, gray1)
	text = pytesseract.image_to_string(Image.open(path1+'/'+filename))
	img = Image.open(path1+'/'+filename)
	img_data = pytesseract.image_to_data(path1+'/'+filename, output_type='dict')
	# print("conf",img_data)
	img_data['conf'] = [int(i) for i in img_data['conf']]
	max_conf = max(img_data["conf"])
	txt = [i for i in img_data['text'] if i is not '']
	if txt:
		confmax.append(max_conf)
	# print(confmax)

	# print("Text per preprocessing",txt)
	if txt:
		textmax.append(txt)
	# print(textmax)
	# img.show()
	# print(text)
	# print(img)
	avg = sum(img_data['conf']) / len(img_data['conf'])
	if txt:
		average_conf.append(avg)
