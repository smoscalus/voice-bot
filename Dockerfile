FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/models/hf

# libopus — кодек голоса Discord (на Windows discord.py везёт его сам, на Linux нужен системный)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libopus0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY bot ./bot

# data — память бота (досье, самоосознание), models — скачанные голоса и Whisper, logs — логи
VOLUME ["/app/data", "/app/models", "/app/logs"]

CMD ["python", "-m", "bot"]
