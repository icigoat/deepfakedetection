# AI Media Detector - Vercel Deployment Guide

## ⚠️ Important Vercel Limitations

Vercel is a **serverless platform** with specific limitations for Django applications:

1. **Read-only filesystem** (except `/tmp`)
2. **50MB deployment size limit**
3. **10-second function timeout** (Hobby plan)
4. **No persistent storage** for uploaded files
5. **Cold starts** may affect performance

### Recommended for Vercel:
- ✅ Demo/prototype deployments
- ✅ Low-traffic applications
- ✅ Testing and development

### NOT Recommended for Vercel:
- ❌ High-traffic production apps
- ❌ Large file processing (videos)
- ❌ Long-running AI analysis tasks

**For production, consider:** Railway, DigitalOcean, AWS, or traditional VPS.

---

## Prerequisites

1. Vercel account (free tier available)
2. GitHub/GitLab repository
3. External database (recommended: Neon, Supabase, or Railway PostgreSQL)
4. External file storage (recommended: AWS S3, Cloudinary, or Vercel Blob)

---

## Step 1: Prepare Your Repository

Ensure these files exist in your repository:
- ✅ `vercel.json` - Vercel configuration
- ✅ `build_files.sh` - Build script
- ✅ `requirements.txt` - Python dependencies
- ✅ `wsgi.py` - WSGI entry point

---

## Step 2: Set Up External Database (Recommended)

### Option A: Neon (PostgreSQL - Free Tier)

1. Go to [neon.tech](https://neon.tech)
2. Create a new project
3. Copy the connection string (looks like: `postgresql://user:pass@host/dbname`)

### Option B: Supabase (PostgreSQL - Free Tier)

1. Go to [supabase.com](https://supabase.com)
2. Create a new project
3. Go to Settings → Database → Connection String
4. Copy the connection string

### Option C: Railway (PostgreSQL)

1. Go to [railway.app](https://railway.app)
2. Create a new PostgreSQL database
3. Copy the `DATABASE_URL`

---

## Step 3: Configure Environment Variables in Vercel

### Via Vercel Dashboard:

1. Go to your project settings
2. Navigate to "Environment Variables"
3. Add the following variables:

```env
# Required
SECRET_KEY=<generate-new-secret-key>
DEBUG=False
ALLOWED_HOSTS=.vercel.app,yourdomain.com

# Database (if using external PostgreSQL)
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Optional - Disable SSL redirect (Vercel handles SSL)
SECURE_SSL_REDIRECT=False
```

### Generate SECRET_KEY:

```python
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

---

## Step 4: Deploy to Vercel

### Method 1: Via Vercel Dashboard (Recommended)

1. Go to [vercel.com/new](https://vercel.com/new)
2. Import your GitHub repository
3. Vercel will auto-detect Django
4. Click "Deploy"

### Method 2: Via Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
vercel --prod
```

---

## Step 5: Run Database Migrations

After first deployment, run migrations:

```bash
# Install Vercel CLI if not already
npm install -g vercel

# Link to your project
vercel link

# Run migrations
vercel exec -- python manage.py migrate

# Create superuser (optional)
vercel exec -- python manage.py createsuperuser
```

---

## Step 6: Configure External File Storage (Important!)

Vercel's filesystem is read-only. You MUST use external storage for uploads.

### Option A: Vercel Blob Storage

1. Install Vercel Blob:
```bash
pip install vercel-blob
```

2. Add to `requirements.txt`:
```
vercel-blob>=0.14.0
```

3. Configure in `settings.py`:
```python
if IS_VERCEL:
    from vercel_blob import put, delete
    # Configure Django to use Vercel Blob
    DEFAULT_FILE_STORAGE = 'storages.backends.vercel_blob.VercelBlobStorage'
```

### Option B: AWS S3 (Recommended for Production)

1. Install django-storages:
```bash
pip install django-storages boto3
```

2. Add environment variables in Vercel:
```env
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_STORAGE_BUCKET_NAME=your-bucket-name
AWS_S3_REGION_NAME=us-east-1
```

3. Update `settings.py`:
```python
if not DEBUG or IS_VERCEL:
    DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
    AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')
    AWS_STORAGE_BUCKET_NAME = config('AWS_STORAGE_BUCKET_NAME')
    AWS_S3_REGION_NAME = config('AWS_S3_REGION_NAME', default='us-east-1')
    AWS_S3_FILE_OVERWRITE = False
    AWS_DEFAULT_ACL = None
```

### Option C: Cloudinary

1. Install cloudinary:
```bash
pip install cloudinary django-cloudinary-storage
```

2. Add environment variables:
```env
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret
```

---

## Step 7: Custom Domain (Optional)

1. Go to your Vercel project settings
2. Navigate to "Domains"
3. Add your custom domain
4. Update DNS records as instructed
5. Update `ALLOWED_HOSTS` in environment variables

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution:** Ensure all dependencies are in `requirements.txt`

```bash
pip freeze > requirements.txt
```

### Issue: Static files not loading

**Solution:** Run collectstatic locally and commit:

```bash
python manage.py collectstatic --noinput
git add staticfiles/
git commit -m "Add static files"
git push
```

### Issue: Database connection errors

**Solution:** Verify `DATABASE_URL` is correct and database is accessible

```bash
# Test connection
vercel exec -- python manage.py dbshell
```

### Issue: Function timeout (10 seconds)

**Solution:** 
- Upgrade to Vercel Pro (60-second timeout)
- Or use a different platform for long-running tasks
- Optimize AI model loading (cache models)

### Issue: Deployment size exceeds 50MB

**Solution:**
- Use `opencv-python-headless` instead of `opencv-python`
- Consider using PyTorch CPU-only version
- Remove unnecessary dependencies

### Issue: Cold starts are slow

**Solution:**
- Upgrade to Vercel Pro for better performance
- Implement model caching
- Use serverless-friendly AI services (AWS Rekognition, Google Vision)

---

## Performance Optimization for Vercel

### 1. Reduce Package Size

Use lightweight versions:
```txt
opencv-python-headless  # Instead of opencv-python
torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only
```

### 2. Cache AI Models

```python
import os
from functools import lru_cache

@lru_cache(maxsize=1)
def get_model():
    # Load model once and cache
    model = load_your_model()
    return model
```

### 3. Use Edge Functions (Beta)

For faster response times, consider Vercel Edge Functions for non-AI routes.

---

## Monitoring

### View Logs

```bash
vercel logs <deployment-url>
```

### Real-time Logs

```bash
vercel logs --follow
```

---

## Cost Considerations

### Vercel Free Tier:
- ✅ 100GB bandwidth/month
- ✅ Unlimited deployments
- ⚠️ 10-second function timeout
- ⚠️ Slower cold starts

### Vercel Pro ($20/month):
- ✅ 1TB bandwidth/month
- ✅ 60-second function timeout
- ✅ Faster cold starts
- ✅ Better performance

---

## Alternative: Hybrid Approach

For best results, consider:

1. **Frontend on Vercel** - Fast, global CDN
2. **Backend on Railway/DigitalOcean** - Better for AI processing
3. **Database on Neon/Supabase** - Managed PostgreSQL
4. **Storage on AWS S3/Cloudinary** - File uploads

---

## Complete Environment Variables Reference

```env
# Django Core
SECRET_KEY=your-secret-key-here
DEBUG=False
ALLOWED_HOSTS=.vercel.app,yourdomain.com

# Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Security
SECURE_SSL_REDIRECT=False

# AWS S3 (if using)
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_STORAGE_BUCKET_NAME=your-bucket
AWS_S3_REGION_NAME=us-east-1

# Cloudinary (if using)
CLOUDINARY_CLOUD_NAME=your-cloud
CLOUDINARY_API_KEY=your-key
CLOUDINARY_API_SECRET=your-secret
```

---

## Support

For Vercel-specific issues:
- [Vercel Documentation](https://vercel.com/docs)
- [Vercel Community](https://github.com/vercel/vercel/discussions)

For Django deployment issues:
- Check `DEPLOYMENT.md` for general Django deployment
- Review Vercel logs: `vercel logs`

---

## Conclusion

Vercel can work for demos and low-traffic apps, but for production AI applications with video processing, consider platforms designed for long-running tasks like Railway, DigitalOcean, or AWS.
