# Main.py

import cv2
import numpy as np
import os

import DetectChars
import DetectPlates
import PossiblePlate

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

###################################################################################################
def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # attempt KNN training

    if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
        print("\nerror: KNN traning was not successful\n")  # show error message
        return                                                          # and exit program
    # end if
    for filename in os.listdir("input5/"):
        print(filename)
        imgOriginalScene  = cv2.imread("input5/"+filename)# open image

        if imgOriginalScene is None:                            # if image was not read successfully
            print("\nerror: image not read from file \n\n")  # print error message to std out
            os.system("pause")                                  # pause so user can see error message
            continue                                              # and exit program
        # end if

        listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates

        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

        #cv2.imshow("imgOriginalScene", imgOriginalScene)            # show scene image

        if len(listOfPossiblePlates) == 0:                          # if no plates were found
            print("\nno license plates were detected\n")  # inform user no plates were found
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
            cv2.imwrite(os.path.join(path, filename), licPlate.imgPlate)
            #cv2.imshow("imgThresh", licPlate.imgThresh)
            #path = 'output_thresh1/'
            #cv2.imwrite(os.path.join(path, "one_"+filename), licPlate.imgThresh)

            if len(licPlate.strChars) == 0:                     # if no chars were found in the plate
                print("\nno characters were detected\n\n")  # show message
                continue                                    # and exit program
            # end if

            drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # draw red rectangle around plate

            print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # write license plate text to std out
            print("----------------------------------------")

            #writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # write license plate text on the image

            #cv2.imshow("imgOriginalScene", imgOriginalScene)                # re-show scene image

            cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           # write image out to file

        # end if else

        #cv2.waitKey(0)					# hold windows open until user presses a key

    return
# end main


def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function

if __name__ == "__main__":
    main()


















