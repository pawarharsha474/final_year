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




def set_damage_claim(new_claim_application_id,list_damage_parts_identfied_by_model,list_additiona_damage_parts_identfied_by_user,list_non_damage_parts_identfied_by_user):
	request_data_json = {}
	request_data_json.update({'db_status': 1})
	request_data_json.update({'db_msg': "Successfully updated to database"})
	request_data_json.update({'new_claim_application_id': new_claim_application_id})
	request_data_json.update({'damage_parts_identfied_by_model': list_damage_parts_identfied_by_model})
	request_data_json.update(
		{'additiona_damage_parts_identfied_by_user': list_additiona_damage_parts_identfied_by_user})
	request_data_json.update({'non_damage_parts_identfied_by_user': list_non_damage_parts_identfied_by_user})

	response = json.dumps(request_data_json)
	return response