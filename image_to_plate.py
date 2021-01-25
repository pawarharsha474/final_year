from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
from flask import Flask, redirect, url_for, request, render_template,send_file
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import DetectChars
import DetectPlates
import PossiblePlate

app = Flask(__name__ , template_folder = "C:/Analysis/templates/")

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
    
    
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'input5', secure_filename(f.filename))
        f.save(file_path)
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # attempt KNN training

    if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
        print("\nerror: KNN traning was not successful\n")  # show error message
        return                                                          # and exit program
    # end if
    for filename in os.listdir("input5/"):
        print(filename)
        imgOriginalScene  = cv2.imread("input5/"+filename)# open image

        if imgOriginalScene is None:                            # if image was not read successfully
            #return("\nerror: image not read from file \n\n") # print error message to std out
            return ("image not read from file")
            os.system("pause")                                  # pause so user can see error message
            continue                                              # and exit program
        # end if

        listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates

        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

        #cv2.imshow("imgOriginalScene", imgOriginalScene)            # show scene image

        if len(listOfPossiblePlates) == 0:                          # if no plates were found
            return ("no license plates were detected")
            #return("\nno license plates were detected\n")  # inform user no plates were found
            #path = 'output5'
            #cv2.imwrite(os.path.join(path, filename), imgOriginalScene)
        else:                                                       # else
                    # if we get in here list of possible plates has at leat one plate

                    # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
            listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                    # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
            licPlate = listOfPossiblePlates[0]

            #cv2.imshow("imgPlate", licPlate.imgPlate)           # show crop of plate and threshold of plate
            path = 'output5/'
            #cv2.imwrite(os.path.join(path, filename), licPlate.imgPlate)
            #return send_file(filename, mimetype='image/gif')
            #cv2.imshow("imgThresh", licPlate.imgThresh)
            #path = 'output_thresh1/'
            #cv2.imwrite(os.path.join(path, "one_"+filename), licPlate.imgThresh)

            if len(licPlate.strChars) == 0:  # if no chars were found in the plate
                return ("no characters were detected")
                #return("\nno characters were detected\n\n")  # show message
                continue                                    # and exit program
            # end if

            drawRedRectangleAroundPlate(imgOriginalScene, licPlate)   # draw red rectangle around plate
            result = "license plate read from image"

            #return("\nlicense plate read from image = " + licPlate.strChars + "\n")  # write license plate text to std out
            #print("----------------------------------------")

            #writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # write license plate text on the image

            #cv2.imshow("imgOriginalScene", imgOriginalScene)                # re-show scene image

            cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           # write image out to file

        # end if else

        #cv2.waitKey(0)					# hold windows open until user presses a key
        return result

    return None
# end main


def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function

if __name__ == "__main__":
    app.run(host='10.44.127.19', port=4000)
    

