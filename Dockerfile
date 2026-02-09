# Dockerfile for Hugging Face Spaces deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Create non-root user with uid 1000 (required by HF Spaces)
RUN useradd -m -u 1000 user

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user:user . .

# Create necessary directories with proper permissions
RUN mkdir -p /home/user/media/uploads /home/user/media/visualizations /home/user/logs && \
    chown -R user:user /home/user

# Switch to non-root user BEFORE running migrations
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Run migrations and collectstatic as the user
RUN python manage.py migrate --noinput
RUN python manage.py collectstatic --noinput

# Expose port 7860 (required by HF Spaces)
EXPOSE 7860

# Start gunicorn on port 7860
CMD ["gunicorn", "ai_detector.wsgi:application", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120"]
