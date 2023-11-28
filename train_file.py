import nltk
import numpy as np
import random

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# download nltk packages
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# create lemmatizer
lemmatizer = WordNetLemmatizer()

# load intents file
import json
intents = json.loads(open('intents.json').read())

