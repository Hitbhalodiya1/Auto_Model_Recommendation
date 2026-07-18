FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application files and install dependencies
COPY pyproject.toml .
COPY . .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Create upload directory
RUN mkdir -p /app/uploads/experiments

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
