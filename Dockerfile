FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    adb \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxcb1 \
    libxext6 \
    libxrender1 \
    libsm6 \
    fontconfig \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY crossybot_v2.py .

ENTRYPOINT ["python3", "crossybot_v2.py", "--backend", "pipe"]
