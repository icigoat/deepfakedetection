@echo off
REM Development startup script for AI Media Detector (Windows)

echo.
echo Starting AI Media Detector (Development Mode)
echo ================================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create .env if it doesn't exist
if not exist ".env" (
    echo Creating .env file...
    (
        echo SECRET_KEY=django-insecure-dev-key-change-in-production
        echo DEBUG=True
        echo ALLOWED_HOSTS=localhost,127.0.0.1
        echo USE_POSTGRES=False
        echo SECURE_SSL_REDIRECT=False
    ) > .env
)

REM Create necessary directories
echo Creating directories...
if not exist "media\uploads\" mkdir media\uploads
if not exist "media\visualizations\" mkdir media\visualizations
if not exist "logs\" mkdir logs
if not exist "static\" mkdir static

REM Run migrations
echo Running database migrations...
python manage.py migrate

REM Collect static files
echo Collecting static files...
python manage.py collectstatic --noinput

echo.
echo Setup complete!
echo.
echo Starting development server...
echo Access at: http://localhost:8000
echo Admin at: http://localhost:8000/admin
echo.
python manage.py runserver

pause
