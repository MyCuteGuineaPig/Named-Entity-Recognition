#!/bin/sh
python3 -m venv env
source ./env/bin/activate
mkdir -p ./env/nltk_data
pip install --upgrade pip
pip install -r requirement.txt
python manage.py runserver