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

def set_feedback(new_claim_application_id,user_name,user_feedback):
	dbinfo = pymysql.connect(database="cost_data", user="root", password="root", host="localhost", port=3307)
	flag=0
	sql_info = "UPDATE assessment SET feedback=%s, user_name =%s WHERE id=%s"
	with dbinfo.cursor(pymysql.cursors.DictCursor) as cursor_info:
		if  cursor_info.execute(sql_info, (user_feedback, user_name, new_claim_application_id)):
			flag=1
	userinfo = {}
	if flag==1:
		userinfo.update({'db_status': 1})
		userinfo.update({'db_msg': "Successfully updated to database"})
		userinfo.update({'new_claim_application_id': new_claim_application_id})
		userinfo.update({'user_name': user_name})
		userinfo.update({'feedback': user_feedback})
	else:
		userinfo.update({'db_status': 0})
		userinfo.update({'db_msg': "Sorry info is not updated to the database"})
	dbinfo.commit()
	dbinfo.close()
	response = json.dumps(userinfo)
	return response
