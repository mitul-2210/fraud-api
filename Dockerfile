    FROM python:3.11-slim
    ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080
    WORKDIR /app
    RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
    COPY requirements.txt ./
    RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
    COPY . .
    EXPOSE 8080
    CMD ["uvicorn","api:app","--host","0.0.0.0","--port","8080"]
