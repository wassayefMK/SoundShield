# SoundShield — Audio DeepFake Detection  
SoundShield is an ML system designed to detect AI-generated (fake) audio using classical and deep learning models.  
As deepfake technology becomes more accessible, audio manipulation poses significant risks in misinformation, fraud, and identity spoofing.  
This project builds a reliable pipeline to classify audio as **REAL** or **FAKE** as a part of the Practical Machine Learning Project IT461 course at King Saud University.

## Project Overview
This repository contains:
- Complete preprocessing and feature-engineering pipeline  
- SVM, XGBoost, and CNN training scripts  
- SMOTE balancing for the highly imbalanced dataset  
- Feature extraction logic using previously-generated 26 audio features  
- Model evaluation: accuracy, recall, and learning curves  
- Visual examples of REAL vs FAKE audio waveforms  
- Final comparison of models  

## Dataset  
The project uses the DEEP-VOICE Dataset  from Kaggle: https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition
The dataset contains 64 WAV audio files:  
- **56 FAKE** (AI-converted via RVC)  
- **8 REAL** recordings  
Each audio file is processed to extract:
- Chroma STFT  
- RMS  
- Spectral Centroid  
- Spectral Bandwidth  
- Rolloff, Zero-Crossing Rate  
- 20 MFCC coefficients  

## Preprocessing Pipeline
We did several steps:
1. Noise Reduction: using the `noisereduce` library  
2. Feature Extraction: using the `Librosa` library
3. Scaling & Normalization 
4. Feature Selection (RFE) → Top 15 features retained  
5. Class Balancing (SMOTE) 
   - Real class oversampled  
   - Fake class undersampled  

## Models Used

### 1- SVM (Support Vector Machine)
- GridSearchCV tuning  
- Best kernel: RBF
- Good performance on small datasets  
- Accuracy & Recall: 87.56%

### 2- XGBoost
- Tuned via GridSearchCV (learning rate, depth, regularization)  
- Strong but slightly lower performance  
- Accuracy: 84.17%, recall: 85.32% 

### 3- CNN (Convolutional Neural Network) 
- 2 convolution layers + max pooling  
- Dense layer with dropout  
- Tuned via Keras RandomSearch  
- Achieved the best generalization  
- **Final Accuracy & Recall: 97.06%**

## Key Takeaways
- The dataset was highly imbalanced — without SMOTE, models performed poorly.
- Preprocessing and feature engineering dramatically improve performance.
- Deep learning strongly outperforms classical machine learning
- CNNs generalize better to real-world audio variability.
