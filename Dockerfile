FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# TensorFlow OS dependencies (minimal + compatible with Debian 12/13)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        libhdf5-dev \
        libgomp1 \
        libjpeg62-turbo-dev \
        liblapack-dev \
        libblas-dev \
        gfortran \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9000
CMD ["python", "app.py"]
