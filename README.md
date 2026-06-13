# Inappropriate Content Detection and Classification of YouTube Videos

📺 An AI-powered system for detecting inappropriate YouTube video content using advanced deep learning techniques. This intelligent system automatically classifies videos as **Safe**, **Violent**, or **Inappropriate**, helping create a safer online environment through automated content moderation and intelligent video screening. 🛡️🚀

This project leverages **EfficientNet-B7** for feature extraction combined with **Bidirectional LSTM (BiLSTM)** for temporal sequence analysis, providing high-accuracy content classification for YouTube videos and other video sources.

---

📋 **Project Overview**

The system integrates multiple deep learning components to create a robust video content classification solution:

- **Automated Content Classification:** Videos are intelligently classified into three categories: Safe, Violent, or Inappropriate.
- **Advanced Feature Extraction:** EfficientNet-B7 neural network extracts rich feature representations from video frames.
- **Temporal Analysis:** Bidirectional LSTM analyzes frame sequences to understand context and temporal patterns.
- **Real-time Prediction:** Frame-by-frame analysis with real-time classification overlay on video output.
- **Comparative Algorithm Analysis:** SVM-based baseline for performance comparison and validation.
- **Comprehensive Metrics:** Precision, recall, F1-score, and accuracy for model evaluation.
- **Interactive GUI:** User-friendly Tkinter-based interface for easy model training and prediction.

---

🔧 **Technical Architecture**

**Deep Learning Models:**
- **EfficientNet-B7:** State-of-the-art CNN for efficient and accurate image feature extraction
- **Bidirectional LSTM (BiLSTM):** Processes temporal sequences from video frames for context-aware classification
- **Support Vector Machine (SVM):** Baseline classifier for comparison (EfficientNet-SVM)

**Key Components:**
- Frame Preprocessing: Grayscale conversion and normalization
- Feature Extraction: EfficientNet-B7 generates 256-dimensional feature vectors
- Sequence Modeling: BiLSTM captures temporal dependencies across frames
- Classification: Softmax output for three-class prediction

---

🗂 **Project Structure**
```
Inappropriate-Content-Detection-and-Classification-of-YouTube-Videos/
├── Main.py                              # Main GUI application
├── model/
│   ├── model_weights.hdf5               # Trained EfficientNet-CNN weights
│   ├── bilstm_weights.hdf5              # Trained BiLSTM model weights
│   ├── content_model.h5                 # Pre-trained content detection model
│   ├── X.txt.npy                        # Preprocessed feature arrays
│   ├── Y.txt.npy                        # Preprocessed labels
│   └── history.pckl                     # Training history
├── Dataset/
│   ├── safe/                            # Safe YouTube video frames
│   └── violence/                        # Violent/Inappropriate video frames
├── testVideos/                          # Test video files for prediction
├── Abstract.docx                        # Project abstract documentation
└── Inappropriate content detection youtube.docx  # Detailed project report
```

---

💡 **Features**

- **Multi-Algorithm Support:** Compare EfficientNet-BiLSTM with EfficientNet-SVM baseline
- **Dataset Preprocessing:** Automatic image resizing, normalization, and shuffling
- **Model Training:** Train models from scratch or load pre-trained weights
- **Real-time Video Analysis:** Process video files with frame-by-frame classification
- **Performance Metrics:** Calculate and display accuracy, precision, recall, and F1-score
- **Confusion Matrix Visualization:** Visual representation of model performance
- **Comparative Analysis:** Side-by-side performance comparison graphs
- **Intuitive GUI:** Tkinter-based interface with clear workflow
- **Checkpoint Saving:** Automatically saves best model weights during training

---

🛠 **Technologies & Libraries**

**Deep Learning Frameworks:**
- TensorFlow/Keras
- EfficientNet (Keras implementation)

**Data Processing:**
- OpenCV (cv2) - Video/Image processing
- NumPy - Numerical computations
- Pandas - Data manipulation
- Scikit-learn - Machine learning utilities

**Visualization:**
- Matplotlib - Graph plotting
- Seaborn - Statistical visualizations

**GUI:**
- Tkinter - User interface development

**ML Metrics:**
- Scikit-learn - Accuracy, precision, recall, F1-score, confusion matrix

---

🚀 **Getting Started**

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- CUDA (optional, for GPU acceleration)
- Required libraries (see requirements)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SyamaVenkatTeja/Inappropriate-Content-Detection-and-Classification-of-YouTube-Videos.git
cd Inappropriate-Content-Detection-and-Classification-of-YouTube-Videos
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Create a `Dataset/` folder with subdirectories: `safe/` and `violence/`
   - Place labeled video frame images in respective directories
   - Recommended: 1000+ images per category for optimal performance

4. Run the application:
```bash
python Main.py
```

### Usage Workflow

1. **Upload Dataset:** Click "Upload Youtube Normal & Inappropriate Content Dataset" to load your dataset
2. **Preprocess Data:** Click "Dataset Preprocessing" to process images and view distribution
3. **Train Models:** 
   - Click "Run Propose DL-BILSTM-GRU Algorithm" to train the BiLSTM model
   - Click "Run EfficientNet-SVM Algorithm" to train the SVM baseline
4. **Compare Performance:** Click "Comparison Graph" to view side-by-side metrics
5. **Predict on Videos:** Click "Inappropriate Content Prediction from Test Video" to test on new videos
6. **Exit:** Click "Exit" to close the application

---

✅ **Model Performance**

The system evaluates three key metrics:
- **Accuracy:** Overall correctness of predictions
- **Precision:** Accuracy of positive predictions
- **Recall:** Ability to identify all positive cases
- **F1-Score:** Harmonic mean balancing precision and recall

The **EfficientNet-BiLSTM** model provides superior performance compared to EfficientNet-SVM through:
- Temporal sequence understanding
- Multi-frame context analysis
- Better generalization on video data
- Reduced false positives/negatives

---

📊 **Algorithm Details**

### EfficientNet-B7 Feature Extraction
- Input: Video frames resized to (32, 32) pixels
- Output: 256-dimensional feature vectors
- Pre-trained weights for transfer learning
- Non-trainable backbone for efficient computation

### Bidirectional LSTM Processing
- Input: CNN feature sequences (batch, 16, 16)
- Two LSTM layers with 32 filters each
- Dropout (20%) for regularization
- Output: Softmax probabilities for three classes

### Training Configuration
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Batch Size:** 16 (BiLSTM), 32 (CNN)
- **Epochs:** 20 (BiLSTM), 50 (CNN)
- **Train-Test Split:** 80-20

---

📈 **Results & Visualization**

The system generates:
- Confusion matrices for each algorithm
- Dataset class distribution bar charts
- Comparative performance graphs
- Real-time prediction overlays on video frames

---

🔮 **Future Enhancements**

- Mobile application for iOS/Android deployment
- Real-time YouTube stream analysis
- Multi-language support for content metadata
- Advanced anomaly detection for edge cases
- Cloud-based API deployment
- Improved model compression for edge devices
- Integration with content moderation platforms
- Support for other video platforms (TikTok, Instagram Reels, etc.)
- Custom model fine-tuning interface

---

⚙️ **Configuration**

Key parameters in `Main.py` can be adjusted:
- Image resize dimensions (currently 32×32)
- Train-test split ratio (currently 80-20)
- LSTM filters and layers (currently 32, 2 layers)
- Batch sizes and epochs
- Dropout rates

---

🙏 **Acknowledgments**

This project was developed as a comprehensive deep learning research initiative focusing on content moderation and video classification.

- **Developed by:** Syama Venkat Teja
- **Project Type:** AI/ML Research Project
- **Focus Area:** Video Content Classification & Moderation
- **Techniques:** EfficientNet, BiLSTM, Transfer Learning, Deep Learning

Special thanks to the open-source community for tools and libraries that made this project possible.

---

📝 **License**

This project is open source and available for educational and research purposes.

---

**For Questions or Support:** Please open an issue in the repository or contact the developer.

**Happy Content Moderation! 🛡️**
