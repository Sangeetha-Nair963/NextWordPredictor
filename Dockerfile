FROM python:3.10-slim
WORKDIR /app
ADD . /app
RUN pip3 install --upgrade pip
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt
RUN apt-get update && apt-get upgrade -y
RUN apt-get install espeak -y
RUN apt-get install ffmpeg -y
RUN apt-get install libespeak1 -y
RUN apt-get install alsa-utils -y
RUN apt-get install libportaudio2 -y
RUN apt-get install libportaudiocpp0 -y
RUN apt-get install libsndfile1-dev -y
RUN apt-get install portaudio19-dev -y
RUN apt-get install pulseaudio -y
RUN apt-get install python3-pyaudio -y
EXPOSE 5000
ENV NAME OpentoAll
CMD ["python","main.py"]