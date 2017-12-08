<<<<<<< Updated upstream
FROM python:3.6
=======
<<<<<<< HEAD
FROM kaggle/python:latest
=======
FROM python:3.6
>>>>>>> parent of f912eb4... 
>>>>>>> Stashed changes

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /usr/src/app
CMD ["python", "app.py"]