"""
Management command to clean up old uploaded files and detections
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from detector.models import Detection
import os


class Command(BaseCommand):
    help = 'Clean up old uploaded files and detection records (older than 7 days)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            default=7,
            help='Delete files older than this many days (default: 7)',
        )

    def handle(self, *args, **options):
        days = options['days']
        cutoff_date = timezone.now() - timedelta(days=days)

        # Find old detections
        old_detections = Detection.objects.filter(created_at__lt=cutoff_date)
        count = old_detections.count()

        if count == 0:
            self.stdout.write(self.style.SUCCESS(f'No files older than {days} days found.'))
            return

        # Delete files and records
        deleted_files = 0
        for detection in old_detections:
            # Delete uploaded file
            if detection.file and os.path.exists(detection.file.path):
                os.remove(detection.file.path)
                deleted_files += 1

            # Delete visualization images
            for img_field in [detection.fft_image, detection.ela_image, detection.noise_image]:
                if img_field and os.path.exists(img_field.path):
                    os.remove(img_field.path)
                    deleted_files += 1

        # Delete database records
        old_detections.delete()

        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully deleted {count} detection records and {deleted_files} files older than {days} days.'
            )
        )
