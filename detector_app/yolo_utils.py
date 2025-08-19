# detector_app/yolo_utils.py
import torch
import os
import yaml
import shutil
from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class YOLOTrainer:
    def __init__(self):
        self.yolov5_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov5')
        self.datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trained_models')
        
        # Create directories if they don't exist
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def create_dataset_structure(self, session_id, classes):
        """Create YOLOv5 dataset structure"""
        dataset_path = os.path.join(self.datasets_dir, f"session_{session_id}")
        
        # Create directory structure
        directories = ['images/train', 'images/val', 'labels/train', 'labels/val']
        for dir_path in directories:
            os.makedirs(os.path.join(dataset_path, dir_path), exist_ok=True)
        
        # Create dataset.yaml file
        dataset_yaml = {
            'path': dataset_path,
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(classes),
            'names': classes
        }
        
        yaml_path = os.path.join(dataset_path, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f)
        
        return dataset_path, yaml_path
    
    def prepare_training(self, session_id, classes, image_files, annotation_files, train_ratio=0.8):
        """Prepare dataset for training"""
        dataset_path, yaml_path = self.create_dataset_structure(session_id, classes)
        
        # Split data into train and validation
        split_idx = int(len(image_files) * train_ratio)
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:]
        train_anns = annotation_files[:split_idx]
        val_anns = annotation_files[split_idx:]
        
        # Copy images and annotations to appropriate directories
        for img_path, ann_path in zip(train_images, train_anns):
            shutil.copy(img_path, os.path.join(dataset_path, 'images/train'))
            if ann_path and os.path.exists(ann_path):
                shutil.copy(ann_path, os.path.join(dataset_path, 'labels/train'))
        
        for img_path, ann_path in zip(val_images, val_anns):
            shutil.copy(img_path, os.path.join(dataset_path, 'images/val'))
            if ann_path and os.path.exists(ann_path):
                shutil.copy(ann_path, os.path.join(dataset_path, 'labels/val'))
        
        return dataset_path, yaml_path
    
    def train_model(self, session_id, yaml_path, epochs=50, batch_size=16, image_size=640):
        """Train YOLOv5 model"""
        model_path = os.path.join(self.models_dir, f"session_{session_id}")
        os.makedirs(model_path, exist_ok=True)
        
        # Training command
        cmd = [
            'python', os.path.join(self.yolov5_dir, 'train.py'),
            '--img', str(image_size),
            '--batch', str(batch_size),
            '--epochs', str(epochs),
            '--data', yaml_path,
            '--weights', 'yolov5s.pt',
            '--project', model_path,
            '--name', 'train'
        ]
        
        # Run training
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.yolov5_dir)
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            # Find the best model
            best_model_path = os.path.join(model_path, 'train', 'weights', 'best.pt')
            if os.path.exists(best_model_path):
                return best_model_path, stdout.decode()
            else:
                raise Exception("Training completed but best model not found")
        else:
            raise Exception(f"Training failed: {stderr.decode()}")
    
    def detect_objects(self, model_path, image_path, confidence_threshold=0.5):
        """Detect objects in an image using trained model"""
        # Load model
        model = torch.hub.load(self.yolov5_dir, 'custom', path=model_path, source='local')
        model.conf = confidence_threshold
        
        # Perform detection
        results = model(image_path)
        
        # Process results
        detections = []
        result_image = results.render()[0]  # Get image with bounding boxes
        
        for *box, conf, cls in results.xyxy[0]:
            detections.append({
                'class': results.names[int(cls)],
                'confidence': float(conf),
                'bbox': [float(x) for x in box]
            })
        
        # Convert result image to PIL format
        result_pil = Image.fromarray(result_image)
        
        return detections, result_pil