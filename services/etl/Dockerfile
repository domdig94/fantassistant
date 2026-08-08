FROM docker.io/python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --trusted-host pypi.org  --no-cache-dir -r requirements.txt

COPY . .

CMD ["sleep", "infinity"]
