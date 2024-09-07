from django.db import models

class Upload(models.Model):
    audio_file = models.FileField(upload_to='uploads/')