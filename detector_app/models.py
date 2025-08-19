# detector_app/models.py
from django.db import models
from django.contrib.auth.models import User

class TrainingSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    model_name = models.CharField(max_length=100)
    classes = models.JSONField()  # List of class names
    dataset_path = models.CharField(max_length=255)
    epochs = models.IntegerField(default=50)
    batch_size = models.IntegerField(default=16)
    image_size = models.IntegerField(default=640)
    status = models.CharField(max_length=20, default='pending')  # pending, training, completed, failed
    accuracy = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.model_name} - {self.status}"

class DetectionResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='detection_images/')
    model_used = models.ForeignKey(TrainingSession, on_delete=models.CASCADE)
    result_image = models.ImageField(upload_to='detection_results/', null=True, blank=True)
    detections = models.JSONField()  # List of detected objects with confidence
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Detection {self.id}"