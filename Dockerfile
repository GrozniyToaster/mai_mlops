FROM python:latest

LABEL authors="Arseniy Polyubin, Nedosekov Ivan"

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && export PATH="/root/.local/bin:$PATH" \
    && uv pip install --system -r requirements.txt \
    && rm -rf /root/.cache \
    && apt-get purge -y --auto-remove curl \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

COPY grpc_server.py .
COPY server.py . 
COPY labels.json .
COPY proto /proto

RUN python3 -m grpc_tools.protoc -I/proto --python_out=. --grpc_python_out=. /proto/inference.proto

EXPOSE 8080 9090

ENTRYPOINT ["sh", "-c", "python3 grpc_server.py & python3 server.py"]