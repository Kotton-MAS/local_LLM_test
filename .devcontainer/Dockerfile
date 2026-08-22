FROM python:3.12-slim-bookworm
# uv はマイナーバージョンで固定 (latest は再現性を損なう)
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv

WORKDIR /workspace

RUN apt update && \
    apt install -y curl build-essential git openssh-client zip unzip tmux && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*