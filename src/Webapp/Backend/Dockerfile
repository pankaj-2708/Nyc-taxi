# base image
FROM python:3.12-slim

# setting a working directory
WORKDIR /app

#copying requirements and installing dependeces
COPY requirements_backend.txt .
RUN pip install -r requirements_backend.txt

COPY . .

EXPOSE 8000

CMD ["python","app.py"]