FROM python:3.13-slim

WORKDIR /srv

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY models ./models
RUN uv sync --frozen --no-dev

EXPOSE 8000
ENV FLASK_RUN_PORT=8000

CMD ["uv", "run", "gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "app.main:app"]
