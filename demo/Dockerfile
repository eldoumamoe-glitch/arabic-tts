FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gradio

COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]
