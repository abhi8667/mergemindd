FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "import yaml; yaml.safe_load(open('openenv.yaml'))" || (echo 'openenv.yaml invalid' && exit 1)

EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "visualization/app.py"]
