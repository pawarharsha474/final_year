import cv2
import json
data={}
import pytesseract
from PIL import Image,ImageChops
import os
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
#result=[]

avgconf_text={}
confmax=[]
textmax=[]
average_conf=[]
from flask import Flask, redirect, url_for, request, render_template

app = Flask(__name__ , template_folder = "C:/Analysis/templates/")

@app.route('/', methods=['GET'])
#def index():
#    return render_template('index.html')


#@app.route('/predict', methods=['GET', 'POST'])
def main():
    for filename in os.listdir("C:/Analysis/output5/"):
        print(filename)

    im = Image.open("C:/Analysis/output5/"+filename)

    def trim(im):
        bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)

    #trim(im).show()
    y11=trim(im).save("crop.png")
    image=cv2.imread("crop.png")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    filename = "new_{}".format(filename)
    print(filename)
    img_process(filename, image)
    #print("BINARY IMAGE")
    #img_process(filename,gray1)
    gray2 = cv2.medianBlur(gray, 3)
    #print("GRAY IMAGE")
    img_process(filename,gray2)
    gray3 = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)[1]
    #print("BINARY INVERSION IMAGE")
    img_process(filename, gray3)
    #gray4 = cv2.threshold(gray2,127,255,cv2.THRESH_BINARY)[1]
    #print("GLOBAL THRESHOLDED IMAGE")
    #img_process(filename, gray4)
    gray5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)[1]
    #print("THRESH TRUNCATED IMAGE")
    img_process(filename, gray5)
    #6
    gray6 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)[1]
    #print("THRESH TOZERO IMAGE")
    img_process(filename, gray6)
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
    img_process(filename, gray11)
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
    m=0
    for i in sorted_text:
        s=''
        if i is not '' and i is not ' ' and i is not '  ':
            s=" ".join(str(j) for j in i)

            al_num=''.join(e for e in s if e.isalnum())
            c=''.join(x for x in al_num if not x.islower())
            #print(c)
            for i in range(0,len(c)):
                if c[i].isalpha():
                    data[i+m] = [c[i:]]
                    m=m+1
                    break
    return json.dumps(data)
def img_process(filename,gray1):

    cv2.imwrite(filename, gray1)
    text = pytesseract.image_to_string(Image.open(filename))
    img = Image.open(filename)
    img_data=pytesseract.image_to_data(filename,output_type='dict')
    #print("conf",img_data)
    img_data['conf']=[int(i) for i in img_data['conf']]
    max_conf=max(img_data["conf"])
    txt = [i for i in img_data['text'] if i is not '']
    if txt:
        confmax.append(max_conf)
    #print(confmax)

    #print("Text per preprocessing",txt)
    if txt:
        textmax.append(txt)
    #print(textmax)
    #img.show()
    #print(text)
    #print(img)
    avg = sum(img_data['conf'])/len(img_data['conf'])
    if txt:
        average_conf.append(avg)
if __name__ == "__main__":
    app.debug = True
    app.run(host='10.44.127.19', port=5000)
    
