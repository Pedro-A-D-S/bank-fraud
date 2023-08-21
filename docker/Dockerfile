FROM python:3.10
WORKDIR /serving
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python ./app.py
