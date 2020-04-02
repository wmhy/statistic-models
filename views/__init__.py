

'''from flask import Blueprint
from flask_cors import CORS
from flask import request
import json

from .svm_view import svm_test


svm_api = Blueprint('svm', __name__)
CORS(svm_api, resources=r'/*')'''
from flask import Blueprint
svm_api = Blueprint('/svm', __name__)

