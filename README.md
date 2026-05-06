# Deepfake_Image
🛡️ DeepFake Forensic Investigation Suite
An advanced deep learning-based forensic tool designed to detect manipulated images using a Dual-Branch CNN architecture combined with traditional forensic signals. 
This project specializes in identifying GAN-generated faces and tampered regions using spatial and frequency-domain analysis.

🚀 Key Features
1.Dual-Branch Detection: Combines spatial (RGB) features and frequency (FFT) artifacts for high-accuracy classification.
2.Explainable AI (XAI): Integrated Grad-CAM to visualize exactly which regions of an image the model considers "suspicious."
3.Multi-Signal Forensics: Includes Error Level Analysis (ELA) and Noise Pattern Analysis to detect compression inconsistencies.
4.Frequency Analysis: Real-time Fast Fourier Transform (FFT) spectrum visualization to identify GAN fingerprints.
5.Forensic Reporting: Generate and download a detailed .txt report of the investigation findings.

🧠 Model ArchitectureThe core of this suite is a custom dual-stream neural network designed for robust forensic analysis.
  System WorkflowInput Image: The system accepts a $224 \times 224 \times 3$ RGB image.RGB Branch: Uses an EfficientNet-B0 backbone to extract deep spatial features (vector size: 1280).
  Frequency Branch: Computes the FFT → Log Magnitude to capture periodic artifacts common in AI-generated imagery, processed through specialized Convolutional Layers.
  Channel Attention: A fusion mechanism that weighs the most important features from both branches.Classifier: A final dense layer outputting the Fake Probability.
  Graph TD
    A[Input Image 224x224x3] --> B1[RGB Branch]
    A --> B2[Frequency Branch]
    B1 --> C1[EfficientNet-B0]
    B2 --> C2[FFT -> Log Magnitude]
    C1 --> D1[Feature Vector 1280]
    C2 --> D2[Conv Layers]
    D1 --> E[Channel Attention & Fusion]
    D2 --> E
    E --> F[Classifier]
    F --> G[Fake Probability]

    🔬 Forensic Methodology
  This suite employs a "Defense-in-Depth" strategy by analyzing four distinct digital signals:
  Spatial Attention (Grad-CAM): Identifies where the CNN is looking. In deepfakes, the model often focuses on the boundaries of the face, the eyes, or the mouth where blending artifacts are common.
  Error Level Analysis (ELA): Detects tampering by saving the image at a specific quality level and calculating the difference. Non-uniform bright patches indicate regions with different compression histories (likely "pasted" elements).
  Noise Pattern Analysis: Analyzes the High-Frequency residual noise. Natural images have a consistent "sensor noise" fingerprint, while AI-generated or tampered images show statistical inconsistencies in these patterns.
  Frequency Domain (FFT): GAN-generated images often leave periodic "checkerboard" artifacts. These appear as anomalous bright spots in the frequency spectrum that are invisible to the naked eye.


  🛠️ Tech Stack
  Language: Python
  Deep Learning: PyTorch, Torchvision
  Web Framework: Streamlit
  Computer Vision: OpenCV, PIL
  Data Analysis: NumPy, Matplotlib, Scikit-Learn

  📁 Project Structure
  ├── app.py                # Main Streamlit Dashboard
  ├── model.py              # Dual-Branch Model Architecture
  ├── forensics.py          # ELA & Noise Analysis Functions
  ├── deepfake_model.pth    # Trained Model Weights
  ├── requirements.txt      # Project Dependencies
  └── README.md             # Project Documentation
  
  ⚙️ Installation & Usage
  Clone the Repository:
  git clone https://github.com/your-username/deepfake-forensic-suite.git
  cd deepfake-forensic-suite
  
  Install Dependencies:
  pip install -r requirements.txt
  
  Run the Application:
  streamlit run app.py

  🛠️ Technical Challenges & Solutions
  The "False Positive" Problem: Real photos from social media often trigger "Fake" verdicts due to aggressive platform compression.
  Solution: Implemented an Ensemble Scoring system (60% CNN + 40% Forensic signals) to reduce sensitivity to standard compression artifacts.
  Explainability: Standard CNNs are "black boxes."Solution: Integrated Grad-CAM using OpenCV to provide forensic investigators with a visual heatmap of the evidence.
  Compute Efficiency: Deep learning models can be heavy for web apps.
  Solution: Leveraged EfficientNet-B0 for its high accuracy-to-parameter ratio, ensuring the suite remains responsive on standard hardware.
  
  📊 Performance
  Backbone: EfficientNet-B0
  Input Resolution: $224 \times 224 \times 3$
  Detection Speed: ~2-3 seconds per image (depending on hardware).

  📝 Disclaimer
  This tool is developed as part of a Minor Project at GNA University. While it utilizes state-of-the-art forensic techniques, it should be used as an assistive tool for 
  digital investigation rather than a definitive legal proof.


