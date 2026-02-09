# 🚀 Vercel Quick Start Guide

## 1. Push to GitHub

```bash
git add .
git commit -m "Prepare for Vercel deployment"
git push origin main
```

## 2. Import to Vercel

1. Go to [vercel.com/new](https://vercel.com/new)
2. Click "Import Project"
3. Select your GitHub repository
4. Click "Import"

## 3. Configure Environment Variables

In Vercel Dashboard → Settings → Environment Variables, add:

```env
SECRET_KEY=<generate-with-command-below>
DEBUG=False
ALLOWED_HOSTS=.vercel.app
DATABASE_URL=<your-database-url>
SECURE_SSL_REDIRECT=False
```

**Generate SECRET_KEY:**
```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

## 4. Deploy

Click "Deploy" button in Vercel dashboard.

## 5. Run Migrations

```bash
npm install -g vercel
vercel link
vercel exec -- python manage.py migrate
```

## 6. Done! 🎉

Your app is live at: `https://your-project.vercel.app`

---

## ⚠️ Important Notes

1. **File Storage**: Vercel filesystem is read-only. Use AWS S3 or Cloudinary for uploads.
2. **Database**: Use external PostgreSQL (Neon, Supabase, Railway).
3. **Timeout**: Free tier has 10-second limit. Upgrade to Pro for 60 seconds.
4. **Size Limit**: 50MB deployment size. Use `opencv-python-headless`.

---

## Need Help?

See `VERCEL_DEPLOYMENT.md` for detailed instructions.
