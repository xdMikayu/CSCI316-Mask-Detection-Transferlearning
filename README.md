# Face Mask Detection with Transfer Learning (CSCI316 Project)

This project implements a deep learning pipeline for classifying face mask usage using **Transfer Learning with InceptionV3**. Built for UOWD’s CSCI316 course, it showcases model fine-tuning, image preprocessing, augmentation, and deployment.

## 🔍 Project Overview

- **Task**: Classify images into `mask`, `no_mask`, and `incorrect_mask`.
- **Approach**: Used a pre-trained InceptionV3 model with custom classification layers.
- **Technologies**: TensorFlow, Keras, OpenCV, Python
- **Team Size**: 10 members

## 🧠 Key Features

- Fine-tuned InceptionV3 for 3-class classification
- Cleaned and augmented a face mask dataset (XML-based)
- Real-time training augmentation with `ImageDataGenerator`
- Achieved high accuracy with minimal training time
- Saved trained model to `face_mask_detector.h5`

## 📁 Repository Structure

```
.
├── Code/               # Main code and trained model
├── CleanData/          # Final cleaned image dataset
├── Report.pdf          # Full project report
├── Presentation.pdf    # Project presentation slides
├── README.md
```

## 📊 Results

- Training Accuracy: 92%  
- Validation Accuracy: 88%  
- Test Accuracy: 85%  
- Early Stopping and Dropout used to prevent overfitting

## 📦 Model Deployment

The trained model (`face_mask_detector.h5`) can be loaded for inference using Keras:

```python
from tensorflow.keras.models import load_model
model = load_model('face_mask_detector.h5')
```

## 📷 Dataset

Due to size, the full dataset is not included. You can download it from:
[Google Drive Link]()

## 📽️ Demo Video

[![Watch the Demo](https://img.youtube.com/vi/qn3jjj9sXQY/0.jpg)](https://youtu.be/qn3jjj9sXQY)

## 👨‍💻 Contributors

- Yasin Sheikh  
- Mikaeel Faraz Safdar  
- Haifa Shkhedem  
- Hiba Nasir  
- Bisham Adil Paracha  
- Shaheer Kashif  
- Eman Yahya  
- Saad Musaddiq  
- Rabail Lal  
- Mohammad Shadi


## 🔗 External Resources

- 📦 **Full Project Files (including `.h5` model, cleaned & raw images)**  
[Google Drive Folder](https://drive.google.com/drive/folders/1bK8mriK1w9v2ArJbLFxrfQPQL9XRFLEK?usp=sharing)

- 🎥 **Demo Video**  
[Watch on YouTube](https://www.youtube.com/watch?v=qn3jjj9sXQY)

- 📁 **Original Dataset**  
[Face Mask Detection Dataset – Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection?resource=download)
