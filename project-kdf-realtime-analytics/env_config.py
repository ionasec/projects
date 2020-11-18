from os import environ, path
from dotenv import load_dotenv


# Find .env file
basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, 'global.env'))


# General Config
SECRET_KEY = environ.get('SECRET_KEY')
FLASK_APP = environ.get('FLASK_APP')
FLASK_ENV = environ.get('FLASK_ENV')

print(FLASK_APP)