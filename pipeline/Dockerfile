
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# System dependencies for Python packages and common data formats.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies separately to improve Docker layer caching.
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy pipeline scripts, dashboard, configuration, and shared modules.
COPY . /app/

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Cloud Run Jobs can override these defaults per job.
# Cloud Run Service deployment overrides this for Streamlit.
ENTRYPOINT ["python"]
CMD ["sample.py"]