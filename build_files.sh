#!/bin/bash

echo "Building AI Media Detector for Vercel..."

# Install dependencies
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --noinput --clear

echo "Build complete!"
