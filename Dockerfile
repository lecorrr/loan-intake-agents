FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LOAN_AGENTS_HOME=/app

# DejaVu draws the Cyrillic text of the synthetic certificates (`loan-agents make-samples`).
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY config ./config
COPY templates ./templates
COPY samples ./samples

RUN useradd --create-home --uid 10001 app && mkdir -p /app/data && chown app:app /app/data
USER app

EXPOSE 8000
CMD ["loan-agents", "serve", "--host", "127.0.0.1", "--port", "8000"]
