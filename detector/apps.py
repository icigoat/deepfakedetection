from django.apps import AppConfig
import os


class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'

    def ready(self):
        # Start cleanup scheduler when app starts
        # Only run in main process (not in reloader during development)
        if os.environ.get('RUN_MAIN', None) != 'true':
            from detector.cleanup_scheduler import start_cleanup_scheduler
            start_cleanup_scheduler()
