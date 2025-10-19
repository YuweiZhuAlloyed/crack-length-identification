FROM mcr.microsoft.com/vscode/devcontainers/python:3.11

WORKDIR /app

COPY requirements.txt .
COPY sam-hq sam-hq

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install -e sam-hq

RUN apt update && apt install -y libgl1
