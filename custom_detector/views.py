# detector_app/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.conf import settings
import os
import json
from .models import TrainingSession, DetectionResult
from .yolo_utils import YOLOTrainer
from django.core.files.base import ContentFile
import tempfile

@login_required
def home(request):
    """Home page with training and detection options"""
    sessions = TrainingSession.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'home.html', {'sessions': sessions})

@login_required
def train_model(request):
    """Train a new object detection model"""
    if request.method == 'POST':
        try:
            # Get training parameters
            model_name = request.POST.get('model_name')
            classes = json.loads(request.POST.get('classes', '[]'))
            epochs = int(request.POST.get('epochs', 50))
            batch_size = int(request.POST.get('batch_size', 16))
            image_size = int(request.POST.get('image_size', 640))
            
            # Get uploaded files
            image_files = request.FILES.getlist('images')
            annotation_files = request.FILES.getlist('annotations')
            
            # Save files temporarily
            temp_dir = tempfile.mkdtemp()
            image_paths = []
            annotation_paths = []
            
            for i, img_file in enumerate(image_files):
                img_path = os.path.join(temp_dir, img_file.name)
                with open(img_path, 'wb+') as f:
                    for chunk in img_file.chunks():
                        f.write(chunk)
                image_paths.append(img_path)
            
            for i, ann_file in enumerate(annotation_files):
                ann_path = os.path.join(temp_dir, ann_file.name)
                with open(ann_path, 'wb+') as f:
                    for chunk in ann_file.chunks():
                        f.write(chunk)
                annotation_paths.append(ann_path)
            
            # Create training session
            session = TrainingSession.objects.create(
                user=request.user,
                model_name=model_name,
                classes=classes,
                dataset_path=temp_dir,
                epochs=epochs,
                batch_size=batch_size,
                image_size=image_size,
                status='training'
            )
            
            # Start training in background (in production, use Celery)
            trainer = YOLOTrainer()
            dataset_path, yaml_path = trainer.prepare_training(
                session.id, classes, image_paths, annotation_paths
            )
            
            # Update session with dataset path
            session.dataset_path = dataset_path
            session.save()
            
            # Train model
            try:
                model_path, training_log = trainer.train_model(
                    session.id, yaml_path, epochs, batch_size, image_size
                )
                session.status = 'completed'
                session.save()
                
                return JsonResponse({
                    'success': True,
                    'session_id': session.id,
                    'message': 'Training completed successfully'
                })
            except Exception as e:
                session.status = 'failed'
                session.save()
                return JsonResponse({
                    'success': False,
                    'error': str(e)
                })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def detect_objects(request):
    """Detect objects in an uploaded image"""
    if request.method == 'POST':
        try:
            session_id = request.POST.get('session_id')
            image_file = request.FILES.get('image')
            confidence = float(request.POST.get('confidence', 0.5))
            
            # Get training session
            session = TrainingSession.objects.get(id=session_id, user=request.user)
            
            # Save uploaded image temporarily
            temp_img_path = os.path.join(tempfile.gettempdir(), image_file.name)
            with open(temp_img_path, 'wb+') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)
            
            # Perform detection
            trainer = YOLOTrainer()
            model_path = os.path.join(trainer.models_dir, f"session_{session_id}", "train", "weights", "best.pt")
            
            if not os.path.exists(model_path):
                return JsonResponse({
                    'success': False,
                    'error': 'Model not found. Please train the model first.'
                })
            
            detections, result_image = trainer.detect_objects(model_path, temp_img_path, confidence)
            
            # Save detection result
            detection = DetectionResult.objects.create(
                user=request.user,
                model_used=session,
                detections=detections
            )
            
            # Save uploaded image
            detection.image.save(image_file.name, ContentFile(image_file.read()))
            
            # Save result image
            img_buffer = ContentFile(b'')
            result_image.save(img_buffer, format='JPEG')
            detection.result_image.save(f'result_{detection.id}.jpg', img_buffer)
            
            return JsonResponse({
                'success': True,
                'detection_id': detection.id,
                'detections': detections,
                'result_url': detection.result_image.url
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def detection_results(request, detection_id):
    """View detection results"""
    detection = DetectionResult.objects.get(id=detection_id, user=request.user)
    return render(request, 'detection_result.html', {'detection': detection})

@login_required
def training_status(request, session_id):
    """Check training status"""
    session = TrainingSession.objects.get(id=session_id, user=request.user)
    return JsonResponse({
        'status': session.status,
        'accuracy': session.accuracy
    })