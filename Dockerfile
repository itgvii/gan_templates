FROM python:3.12.11-slim-bookworm

SHELL ["/bin/bash", "-c"]

# set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt -y update && apt -y install curl

# установка Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry POETRY_VERSION=2.1.3 python3 -

ENV PATH="/opt/poetry/bin:$PATH"

COPY poetry.lock pyproject.toml ./
RUN poetry install --without dev

RUN mkdir app
COPY app/coolzombie.png app/main.py app/
RUN mkdir src
COPY src/onnx_generator.py src/
RUN mkdir models_onnx
COPY models_onnx/wgan_1.onnx models_onnx/
COPY .root_anchor ./
# RUN ls -la && sleep 3600

# CMD ["uvicorn", "main:app", "--reload"]

CMD ["poetry", "run", "gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "--workers", "4", "-b", "0.0.0.0:8000"]