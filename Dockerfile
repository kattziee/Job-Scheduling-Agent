FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment files
COPY job_scheduling_env.py .
COPY task_graders.py .
COPY baseline_inference.py .
COPY openenv.yaml .

# Create results directory
RUN mkdir -p /app/results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run baseline by default
CMD ["python", "baseline_inference.py"]
