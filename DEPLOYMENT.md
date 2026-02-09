# AI Media Detector - Production Deployment Guide

## Prerequisites

- Python 3.10+
- PostgreSQL (recommended for production)
- Domain name with SSL certificate
- Server with at least 2GB RAM (4GB+ recommended for PyTorch)

## Environment Setup

### 1. Clone and Install Dependencies

```bash
git clone <your-repo-url>
cd deepfakedetection
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your production values:

```env
SECRET_KEY=<generate-new-secret-key>
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

USE_POSTGRES=True
DB_NAME=ai_detector_db
DB_USER=your_db_user
DB_PASSWORD=your_secure_password
DB_HOST=localhost
DB_PORT=5432

SECURE_SSL_REDIRECT=True
```

**Generate a new SECRET_KEY:**
```python
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

### 3. Database Setup

**PostgreSQL (Recommended):**

```bash
# Create database
sudo -u postgres psql
CREATE DATABASE ai_detector_db;
CREATE USER your_db_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE ai_detector_db TO your_db_user;
\q

# Run migrations
python manage.py migrate
python manage.py createsuperuser
```

**SQLite (Development only):**

Set `USE_POSTGRES=False` in `.env`

### 4. Collect Static Files

```bash
python manage.py collectstatic --noinput
```

### 5. Create Required Directories

```bash
mkdir -p media/uploads media/visualizations logs
```

## Deployment Options

### Option 1: Gunicorn + Nginx (VPS/Dedicated Server)

#### Install Nginx

```bash
sudo apt update
sudo apt install nginx
```

#### Create Gunicorn Service

Create `/etc/systemd/system/ai-detector.service`:

```ini
[Unit]
Description=AI Media Detector Gunicorn Service
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/path/to/deepfakedetection
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/gunicorn \
    --workers 3 \
    --timeout 120 \
    --bind unix:/path/to/deepfakedetection/gunicorn.sock \
    ai_detector.wsgi:application

[Install]
WantedBy=multi-user.target
```

#### Configure Nginx

Create `/etc/nginx/sites-available/ai-detector`:

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    client_max_body_size 50M;

    location = /favicon.ico { access_log off; log_not_found off; }
    
    location /static/ {
        alias /path/to/deepfakedetection/staticfiles/;
    }

    location /media/ {
        alias /path/to/deepfakedetection/media/;
    }

    location / {
        include proxy_params;
        proxy_pass http://unix:/path/to/deepfakedetection/gunicorn.sock;
        proxy_read_timeout 120s;
    }
}
```

Enable site and restart services:

```bash
sudo ln -s /etc/nginx/sites-available/ai-detector /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable ai-detector
sudo systemctl start ai-detector
```

#### SSL with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

### Option 2: Heroku

```bash
# Install Heroku CLI
heroku login
heroku create your-app-name

# Add PostgreSQL
heroku addons:create heroku-postgresql:mini

# Set environment variables
heroku config:set SECRET_KEY=<your-secret-key>
heroku config:set DEBUG=False
heroku config:set USE_POSTGRES=True

# Deploy
git push heroku main

# Run migrations
heroku run python manage.py migrate
heroku run python manage.py createsuperuser
```

### Option 3: Railway

1. Connect your GitHub repository to Railway
2. Add PostgreSQL database
3. Set environment variables in Railway dashboard
4. Deploy automatically on push

### Option 4: DigitalOcean App Platform

1. Create new app from GitHub repository
2. Add managed PostgreSQL database
3. Configure environment variables
4. Set build command: `pip install -r requirements.txt && python manage.py collectstatic --noinput`
5. Set run command: `gunicorn ai_detector.wsgi:application`

## Performance Optimization

### 1. Enable Caching (Redis)

Add to `requirements.txt`:
```
django-redis>=5.4.0
```

Add to `settings.py`:
```python
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': config('REDIS_URL', default='redis://127.0.0.1:6379/1'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}
```

### 2. Configure Gunicorn Workers

For CPU-intensive tasks (AI detection):
```bash
workers = (2 * CPU_cores) + 1
```

For 2 CPU cores:
```bash
gunicorn --workers 5 --timeout 120 ai_detector.wsgi:application
```

### 3. Media File Storage (AWS S3)

For production, use cloud storage:

```bash
pip install django-storages boto3
```

Add to `settings.py`:
```python
if not DEBUG:
    DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
    AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')
    AWS_STORAGE_BUCKET_NAME = config('AWS_STORAGE_BUCKET_NAME')
    AWS_S3_REGION_NAME = config('AWS_S3_REGION_NAME', default='us-east-1')
```

## Monitoring & Maintenance

### 1. Check Logs

```bash
# Application logs
tail -f logs/django.log

# Gunicorn logs
sudo journalctl -u ai-detector -f

# Nginx logs
sudo tail -f /var/log/nginx/error.log
```

### 2. Database Backups

```bash
# PostgreSQL backup
pg_dump -U your_db_user ai_detector_db > backup_$(date +%Y%m%d).sql

# Restore
psql -U your_db_user ai_detector_db < backup_20260208.sql
```

### 3. Media Files Cleanup

Create a cron job to clean old uploads:

```bash
# Add to crontab
0 2 * * * cd /path/to/deepfakedetection && python manage.py cleanup_old_files
```

## Security Checklist

- [ ] `DEBUG=False` in production
- [ ] Strong `SECRET_KEY` generated
- [ ] `ALLOWED_HOSTS` configured correctly
- [ ] SSL certificate installed
- [ ] Database password is strong
- [ ] File upload limits configured
- [ ] CSRF protection enabled
- [ ] Security headers configured
- [ ] Regular backups scheduled
- [ ] Monitoring and logging enabled

## Troubleshooting

### Issue: Static files not loading

```bash
python manage.py collectstatic --clear --noinput
sudo systemctl restart ai-detector
```

### Issue: 502 Bad Gateway

Check Gunicorn is running:
```bash
sudo systemctl status ai-detector
sudo journalctl -u ai-detector -n 50
```

### Issue: Upload fails

Check file permissions:
```bash
sudo chown -R www-data:www-data media/
sudo chmod -R 755 media/
```

### Issue: Out of memory

Reduce Gunicorn workers or upgrade server RAM. PyTorch models require significant memory.

## Support

For issues and questions, check the main README.md or create an issue on GitHub.
