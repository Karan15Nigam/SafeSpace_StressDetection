# SafeSpace_StressDetection
Hugging Face Deployment:https://huggingface.co/spaces/Karan15Nigam/safespace_microexpression/tree/main

SafeSpace leverages state-of-the-art machine learning and deep learning techniques to detect user stress levels through real-time facial expression and microexpression analysis. The system is built for robust, accurate classification of emotional states, providing the foundation for responsive stress management and feedback applications.

#Technical Overview
->Deep Learning Models:
1)ResNet50: Utilized as a feature extractor for high-level facial representations. Pretrained on large-scale face datasets, fine-tuned for emotion recognition on FER+ and CK+ datasets.
2)Vision Transformers (ViT): Integrated for capturing global dependencies in facial images, enabling improved generalization and robustness over classical CNNs.

->Extreme Learning Machine (ELM):
1)A lightweight, single-layer feedforward neural network trained on features extracted from ResNet50 and ViT models, enabling fast, real-time emotion and microexpression classification.

->Microexpression Analysis:
1)Microexpression tracking is performed using Mediapipe FaceMesh, extracting fine-grained facial landmarks for subtle emotion cues (e.g., eye openness, lip movements).
2)Custom calibration routines allow user-specific baseline adaptation.

->Engagement & Stress Classification:
1)Emotions and microexpressions are mapped to engagement and stress categories using a data-driven mapping (e.g., Engaged, Partially Engaged, Not Engaged, Stressed).
2)Real-time prediction pipeline combines macro (emotion) and micro (expression) features for holistic stress inference.

->Technologies Used:
1)Python (TensorFlow, PyTorch, scikit-learn, OpenCV, Mediapipe, Transformers)
2)Pretrained Models: ResNet50, ViT from Hugging Face
3)Real-time Video Processing: OpenCV, Mediapipe

