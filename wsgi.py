"""
WSGI config for Vercel deployment
"""
from ai_detector.wsgi import application

# Vercel serverless function handler
app = application
