FROM python:3.10.7
COPY . /FBHP_app
WORKDIR /FBHP_app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app
