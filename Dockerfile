# Simple Dockerfile for running the QCL demo and tests
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./

# Install system deps that help scientific Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY . .

# Default command prints help for the CLI
CMD ["python", "app/main.py", "--help"]
