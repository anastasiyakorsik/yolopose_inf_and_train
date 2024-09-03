FROM python:3.10

#RUN pip install --upgrade pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /

COPY . /

RUN pip3 install -r /src/requirements.txt

ENTRYPOINT ["python","/src/main.py"]