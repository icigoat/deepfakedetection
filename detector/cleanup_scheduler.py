"""
Automatic cleanup scheduler - runs every 5 minutes
Deletes all media files and flushes database
"""
import threading
import time
import os
import shutil
from django.core.management import call_command
from django.conf import settings
from django.utils import timezone
from datetime import timedelta


def cleanup_task():
    """Run cleanup every 5 minutes"""
    while True:
        try:
            # Wait 5 minutes first (so it doesn't run immediately on startup)
            time.sleep(300)  # 300 seconds = 5 minutes
            
            print(f"[{timezone.now()}] Running automatic cleanup...")
            
            # 1. Delete all media files
            media_root = settings.MEDIA_ROOT
            if os.path.exists(media_root):
                for folder in ['uploads', 'visualizations']:
                    folder_path = os.path.join(media_root, folder)
                    if os.path.exists(folder_path):
                        # Count files before deletion
                        file_count = sum([len(files) for _, _, files in os.walk(folder_path)])
                        
                        # Delete and recreate folder
                        shutil.rmtree(folder_path)
                        os.makedirs(folder_path, exist_ok=True)
                        
                        print(f"  ✓ Deleted {file_count} files from {folder}/")
            
            # 2. Flush database (delete all records)
            call_command('flush', '--noinput')
            print("  ✓ Database flushed")
            
            print(f"[{timezone.now()}] Cleanup completed successfully\n")
            
        except Exception as e:
            print(f"[{timezone.now()}] Cleanup error: {e}\n")


def start_cleanup_scheduler():
    """Start the cleanup scheduler in a background thread"""
    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()
    print("=" * 60)
    print("🧹 Automatic Cleanup Scheduler Started")
    print("   - Runs every 5 minutes")
    print("   - Deletes all media files")
    print("   - Flushes database")
    print("=" * 60 + "\n")
