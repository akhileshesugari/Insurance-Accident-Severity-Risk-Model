# ── Base image ──
FROM python:3.11-slim

# ── Metadata ──
LABEL maintainer="Akhilesh Esugari <akhileshesugari@gmail.com>"
LABEL description="Accident Severity Prediction API"
LABEL version="1.0.0"

# ── Set working directory ──
WORKDIR /app

# ── Copy requirements first (layer caching) ──
COPY requirements.txt .

# ── Install dependencies ──
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application files ──
COPY app.py .
COPY model.pkl .
COPY imputer.pkl .
COPY scaler.pkl .
COPY features.pkl .

# ── Expose port ──
EXPOSE 5000

# ── Health check ──
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# ── Run application ──
CMD ["python", "app.py"]
