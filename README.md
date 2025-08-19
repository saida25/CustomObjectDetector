# Custom Object Detector with Transfer Learning

A Django-based web application that allows you to train custom object detection models using YOLOv5 and TensorFlow with transfer learning. This project enables you to create detectors for your pets, specific objects, or any custom classes.

## Features

- **Custom Model Training**: Train YOLOv5 models on your own dataset
- **Web-based Interface**: Easy-to-use web interface for training and detection
- **Transfer Learning**: Leverages pre-trained YOLOv5 models for faster training
- **Real-time Detection**: Detect objects in uploaded images using your trained models
- **Training Management**: Track and manage multiple training sessions
- **Results Visualization**: View detection results with bounding boxes and confidence scores

## Technology Stack

- **Backend**: Django (Python web framework)
- **Object Detection**: YOLOv5 (PyTorch implementation)
- **Transfer Learning**: Pre-trained YOLOv5 weights
- **Image Processing**: OpenCV, Pillow
- **Frontend**: HTML5, CSS3, JavaScript
- **Database**: SQLite (default, configurable for production)

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- 4GB+ RAM (8GB recommended for training)
- 2GB+ free disk space

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd object_detector
```

### 2. Set Up Virtual Environment

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Clone YOLOv5 Repository

```bash
git clone https://github.com/ultralytics/yolov5.git
```

### 5. Set Up Django Project

```bash
python manage.py makemigrations
python manage.py migrate
```

### 6. Create Superuser (Optional)

```bash
python manage.py createsuperuser
```

### 7. Run the Development Server

```bash
python manage.py runserver
```

### 8. Access the Application

Open your web browser and navigate to:
- Main application: http://localhost:8000
- Admin interface: http://localhost:8000/admin (if you created a superuser)

## Usage

### Training a Custom Model

1. **Navigate to the training section** on the home page
2. **Provide model details**:
   - Model name
   - Class names (comma-separated, e.g., "cat,dog,person")
3. **Upload training data**:
   - Images (JPG, PNG, etc.)
   - Annotation files in YOLO format (optional but recommended)
4. **Adjust training parameters** (optional):
   - Epochs (default: 50)
   - Batch size (default: 16)
5. **Start training** - the process may take several minutes to hours depending on your dataset size

### Detecting Objects

1. **Select a trained model** from the dropdown menu
2. **Upload an image** to analyze
3. **Adjust confidence threshold** (0-1, default: 0.5)
4. **Run detection** - results will show detected objects with confidence scores

### Dataset Preparation

For best results, prepare your dataset with the following structure:

```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
```

**YOLO Annotation Format** (each .txt file):
```
<class_id> <center_x> <center_y> <width> <height>
```

Example:
```
0 0.5 0.5 0.2 0.3
1 0.7 0.3 0.1 0.2
```

Where:
- `class_id` is the index of the class (0-based)
- Coordinates are normalized to [0, 1] relative to image dimensions

## Project Structure

```
object_detector/
├── custom_detector/          # Django project settings
├── detector_app/             # Main Django application
│   ├── models.py            # Database models
│   ├── views.py             # Application logic
│   ├── urls.py              # URL routing
│   ├── yolo_utils.py        # YOLO training and detection utilities
│   └── admin.py             # Admin interface configuration
├── yolov5/                  # YOLOv5 implementation (cloned from ultralytics)
├── datasets/                # Training datasets
├── trained_models/          # Saved models
├── media/                   # Uploaded images and results
├── templates/               # HTML templates
│   ├── home.html            # Main interface
│   └── detection_result.html # Results page
├── static/                  # Static files (CSS, JS, images)
├── manage.py                # Django management script
└── requirements.txt         # Python dependencies
```

## Configuration

### Model Parameters

You can adjust training parameters in the web interface or modify default values in `detector_app/yolo_utils.py`:

- `image_size`: Input image dimensions (default: 640)
- `epochs`: Training epochs (default: 50)
- `batch_size`: Batch size (default: 16)
- `confidence_threshold`: Detection confidence (default: 0.5)

### Database

By default, the project uses SQLite. To use another database, update the `DATABASES` setting in `custom_detector/settings.py`.

## Performance Considerations

- **Training Time**: Depends on dataset size, image resolution, and number of epochs
- **Hardware**: GPU acceleration is recommended for faster training (not required)
- **Memory**: Larger batch sizes require more RAM
- **Storage**: Trained models and datasets can consume significant disk space

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'torch'"**
   - Solution: Reinstall PyTorch with correct version: `pip install torch==2.0.1`

2. **"YOLOv5 directory not found"**
   - Solution: Ensure you've cloned the YOLOv5 repository: `git clone https://github.com/ultralytics/yolov5.git`

3. **Training fails with memory error**
   - Solution: Reduce batch size or image dimensions

4. **Detection results are poor**
   - Solution: Improve training data quality, add more examples, or increase training epochs

### Getting Help

1. Check the YOLOv5 documentation: https://github.com/ultralytics/yolov5
2. Review Django documentation: https://docs.djangoproject.com/
3. Check issues in the project repository

## Production Deployment

For production deployment, consider:

1. Setting `DEBUG = False` in `settings.py`
2. Using a production database (PostgreSQL recommended)
3. Configuring a production web server (Gunicorn + Nginx)
4. Implementing task queue for training (Celery + Redis)
5. Setting up proper static and media file serving
6. Implementing user authentication and authorization
7. Adding rate limiting and security measures

## License

This project uses YOLOv5 which is licensed under the AGPL-3.0 License. Please check the license terms before using in commercial applications.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- Feature requests
- Documentation improvements
- Performance enhancements

## Acknowledgments

- YOLOv5 by Ultralytics (https://github.com/ultralytics/yolov5)
- Django Web Framework (https://www.djangoproject.com/)
- PyTorch Deep Learning Framework (https://pytorch.org/)

## Support

If you encounter issues or have questions:

1. Check the troubleshooting section above
2. Review documentation links provided
3. Search for similar issues in the project repository
4. Create a new issue with detailed information about your problem

---

**Note**: This implementation is designed for educational and demonstration purposes. For production use, consider implementing additional security measures, error handling, and performance optimizations.