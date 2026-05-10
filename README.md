🛡️ DeepFake Forensic Investigation Suite

Advanced 6-Signal Telemetry: CNN + ELA + Noise + DCT + Texture + Edge


An advanced, multi-layered forensic tool designed to detect AI-generated manipulations and digital tampering. Unlike standard detectors, this suite uses an Ensemble Evidence approach, allowing traditional forensic signals to override the AI if specific manipulation traces are found.

https://rupa49-deepfakeimage.streamlit.app/


🚀 What's New: The 6-Signal Engine

The system now runs 6 independent forensic signals simultaneously to provide a comprehensive analysis:


1.CNN Face-Swap Detector: Deep learning analysis trained on the CelebsV2 dataset to find structural facial inconsistencies.


2.Error Level Analysis (ELA): Identifies different compression levels within a single frame—perfect for spotting "pasted" or "cloned" objects.


3.Sensor Noise Maps: Extracts high-frequency residual noise to check for sensor fingerprint consistency.


4.DCT Frequency Analysis: Scans for "checkerboard" artifacts and GAN fingerprints in the Discrete Cosine Transform domain.


5.Texture Uniformity (LBP): Uses Local Binary Patterns to find unnatural smoothness or "robotic" skin textures common in AI generation.


6.Edge Sharpness Profiles: Analyzes the transition gradients of edges to detect blurring or blending used to hide manipulation boundaries.


🧠 Technical Architecture

The core remains a Dual-Branch CNN, but it is now supported by an ensemble logic:


1.RGB Branch: EfficientNet-B0 backbone extracting spatial features.

2.Frequency Branch: FFT-based log-magnitude analysis.

3.Smart Ensemble Algorithm: A weighting mechanism that prevents "False Positives" from high compression by cross-referencing CNN results with traditional Texture and Edge data.

🔬 Detection Capabilities

Face-swap deepfakes: Detects identity replacement.

1.AI-generated images: Identifies traces from Midjourney, DALL-E, and Stable Diffusion.

2.Image manipulations: Spots pasted objects or clone-stamping.

3.Compression artifacts: Differentiates between a real re-saved JPEG and a manipulated one.


🛠️ Tech Stack

 Core: Python, PyTorch

 UI/UX: Streamlit (Custom CSS for Dark Forensic Theme)

 Forensics: OpenCV, Scikit-Image, Plotly (for interactive signal graphs)

 Deployment: GitHub & Streamlit Cloud


📁 Project Structure

├── app.py              # New 6-Signal Dashboard UI
├── model.py            # Dual-Branch CNN (EfficientNet-B0 + FFT)
├── forensics.py        # ELA, DCT, LBP, and Edge analysis logic
├── deepfake_model.pth  # Trained weights (CelebsV2)
├── requirements.txt    # Now includes Plotly and Scikit-Image
└── README.md           # Documentation


⚙️ Installation & Local Usage

# 1. Clone the repo
git clone https://github.com/Rupa49/deepfake_Image.git

# 2. Install requirements (Ensure Plotly is included)
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py


📝 Disclaimer
This tool was developed as a forensic project at GNA University. It is designed to assist digital investigators and should be used as a high-probability detection tool rather than absolute legal evidence.