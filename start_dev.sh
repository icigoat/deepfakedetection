#!/bin/bash

# Development startup script for AI Media Detector

echo "🚀 Starting AI Media Detector (Development Mode)"
echo "================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cat > .env << EOF
SECRET_KEY=django-insecure-dev-key-change-in-production
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
USE_POSTGRES=False
SECURE_SSL_REDIRECT=False
EOF
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p media/uploads media/visualizations logs static

# Run migrations
echo "🗄️  Running database migrations..."
python manage.py migrate

# Collect static files
echo "📦 Collecting static files..."
python manage.py collectstatic --noinput

# Create superuser if needed
echo ""
echo "👤 Create a superuser account (optional):"
python manage.py createsuperuser --noinput --username admin --email admin@example.com 2>/dev/null || echo "Superuser already exists or skipped"

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 Starting development server..."
echo "   Access at: http://localhost:8000"
echo "   Admin at: http://localhost:8000/admin"
echo ""
python manage.py runserver
