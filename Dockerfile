FROM python:3.9

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src ./src
COPY notebooks ./notebooks
COPY README.md .

# Create outputs directory
RUN mkdir -p /app/outputs

# MLflow tracking
ENV MLFLOW_TRACKING_URI=file:///app/outputs/mlruns

CMD ["python", "-m", "src.optimize"]
