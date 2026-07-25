FROM python:3.13-slim

WORKDIR /srv

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY models ./models
RUN uv sync --frozen --no-dev

EXPOSE 8000
ENV PORT=8000
ENV WEB_CONCURRENCY=1

CMD ["sh", "-c", "uv run gunicorn --bind 0.0.0.0:${PORT} --workers ${WEB_CONCURRENCY} --timeout 120 app.main:app"]
