FROM python:3.5
RUN mkdir -p /usr/src/app
RUN mkdir -p /usr/src/output
WORKDIR /usr/src/app
VOLUME ["/usr/src/output"]

COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /usr/src/app
CMD ["python", "app.py"]
