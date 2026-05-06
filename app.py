"""
app.py  –  AI Deepfake Forensic Investigation Suite
─────────────────────────────────────────────────────────────────────────────
Unique features added over the original:
  1. Multi-signal forensic panel  (ELA + Noise Map + Metadata)
  2. Frequency-domain (FFT) spectrum visualisation
  3. Dual-branch model (RGB + Frequency) via model.py
  4. Combined Threat Score  (CNN + forensic ensemble)
  5. Animated confidence gauge
  6. Forensic report download  (text summary)
─────────────────────────────────────────────────────────────────────────────
"""
import io
import datetime
import numpy as np
import cv2
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

from model     import DeepfakeDetector
from forensics import run_full_forensics


# ─────────────────────────────────────────────────────────────────────────────
# Page Config & CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DeepFake Forensic Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Dark forensic theme */
[data-testid="stAppViewContainer"] {
    background: #0d0f14;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: #131720;
    border-right: 1px solid #1e2535;
}
.metric-card {
    background: #161c2a;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.verdict-fake {
    font-size: 2.4rem;
    font-weight: 800;
    color: #ff4b6e;
    text-shadow: 0 0 20px rgba(255,75,110,0.4);
    letter-spacing: 0.08em;
}
.verdict-real {
    font-size: 2.4rem;
    font-weight: 800;
    color: #00e5a0;
    text-shadow: 0 0 20px rgba(0,229,160,0.4);
    letter-spacing: 0.08em;
}
.section-header {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: #4a90d9;
    text-transform: uppercase;
    margin: 18px 0 8px 0;
    border-left: 3px solid #4a90d9;
    padding-left: 8px;
}
.flag-item {
    font-size: 0.82rem;
    color: #fbbf24;
    padding: 3px 0;
}
.flag-ok {
    font-size: 0.82rem;
    color: #34d399;
    padding: 3px 0;
}
div[data-testid="metric-container"] {
    background: #161c2a;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DeepfakeDetector().to(device)
    try:
        model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
        model.eval()
        status = "✅ Trained weights loaded"
    except FileNotFoundError:
        # Run in demo mode (random weights) so the UI still works
        model.eval()
        status = "⚠️ Demo mode – no trained weights found"
    return model, device, status


model, device, model_status = load_model()


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM Helper
# ─────────────────────────────────────────────────────────────────────────────

def grad_cam(model, input_tensor, original_img_rgb):
    target_layer = model.gradcam_target
    activations, gradients = {}, {}

    h1 = target_layer.register_forward_hook(
        lambda m, i, o: activations.update({"f": o.detach()})
    )
    h2 = target_layer.register_full_backward_hook(
        lambda m, i, o: gradients.update({"b": o[0].detach()})
    )

    output = model(input_tensor)
    model.zero_grad()
    output.backward()

    h1.remove(); h2.remove()

    grads   = gradients["b"]
    fmaps   = activations["f"]
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam     = torch.sum(weights * fmaps, dim=1).squeeze().cpu().numpy()
    cam     = np.maximum(cam, 0)
    cam     = cv2.resize(cam, (original_img_rgb.shape[1], original_img_rgb.shape[0]))
    cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    heatmap     = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay     = cv2.addWeighted(original_img_rgb, 0.55, heatmap_rgb, 0.45, 0)
    return overlay, output


# ─────────────────────────────────────────────────────────────────────────────
# FFT Spectrum Helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_fft_spectrum(pil_img):
    gray = np.array(pil_img.convert("L")).astype(np.float32)
    fft  = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1)
    norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    spectrum_rgb = cv2.applyColorMap(np.uint8(norm * 255), cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(spectrum_rgb, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# Score Bar
# ─────────────────────────────────────────────────────────────────────────────

def score_bar_html(label, value, color="#4a90d9"):
    pct = int(value * 100)
    return f"""
    <div style="margin:6px 0">
      <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#94a3b8;margin-bottom:3px">
        <span>{label}</span><span>{pct}%</span>
      </div>
      <div style="background:#1e2535;border-radius:4px;height:6px">
        <div style="background:{color};width:{pct}%;height:6px;border-radius:4px;
                    transition:width 0.6s ease"></div>
      </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 Forensic Suite")
    st.caption(model_status)
    st.divider()
    st.markdown("### Analysis Settings")
    threshold   = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.01,
                            help="Probability above this → FAKE verdict")
    run_forensic= st.toggle("Run Multi-Signal Forensics", value=True,
                            help="ELA + Noise Map + Metadata analysis")
    show_fft    = st.toggle("Show FFT Spectrum", value=True,
                            help="Frequency-domain artifact visualisation")
    st.divider()
    st.markdown("""
    **Signal Legend**
    - 🔴 Grad-CAM – CNN attention regions  
    - 🟣 ELA – Compression artifacts  
    - 🔵 Noise Map – Sensor noise inconsistency  
    - 🟡 FFT – Frequency GAN fingerprints  
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🛡️ DeepFake Forensic Investigation Suite")
st.caption("Multi-signal AI forensics: CNN + ELA + Noise Analysis + Frequency Domain")
st.divider()

uploaded_file = st.file_uploader(
    "Upload an image for forensic analysis",
    type=["jpg", "jpeg", "png"],
    help="Supports face images, portraits, and social media photos"
)

if uploaded_file is None:
    st.info("Upload an image to begin forensic analysis.", icon="📁")
    st.stop()


# ─── Pre-process ──────────────────────────────────────────────────────────────
file_bytes   = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
original_img = cv2.imdecode(file_bytes, 1)
img_rgb      = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
pil_img      = Image.fromarray(img_rgb)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = transform(pil_img).unsqueeze(0).to(device)

with st.spinner("Running forensic analysis …"):

    # ── CNN + Grad-CAM ──────────────────────────────────────────────
    overlay, output = grad_cam(model, input_tensor, img_rgb)

    prob       = torch.sigmoid(output).item()
    prediction = "FAKE" if prob > threshold else "REAL"
    confidence = prob if prob > threshold else (1 - prob)

    # ── Multi-signal forensics ──────────────────────────────────────
    forensic_data = run_full_forensics(pil_img) if run_forensic else None

    # ── FFT spectrum ────────────────────────────────────────────────
    fft_img = compute_fft_spectrum(pil_img) if show_fft else None

    # ── Combined threat score ───────────────────────────────────────
    cnn_score = prob
    if forensic_data:
        # Ensemble: 60% CNN + 40% forensic signals
        combined  = 0.60 * cnn_score + 0.40 * forensic_data["forensic_score"]
    else:
        combined  = cnn_score
    combined = min(combined, 1.0)
    final_verdict = "FAKE" if combined > threshold else "REAL"


# ═════════════════════════════════════════════════════════════════════════════
# RESULTS LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

col_left, col_mid, col_right = st.columns([1.1, 1.1, 1.1])

# ─── Left: Verdict ───────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-header">Forensic Verdict</div>', unsafe_allow_html=True)
    verdict_cls = "verdict-fake" if final_verdict == "FAKE" else "verdict-real"
    st.markdown(f'<div class="{verdict_cls}">{final_verdict}</div>', unsafe_allow_html=True)

    st.metric("CNN Probability (Fake)", f"{prob*100:.2f}%")
    if forensic_data:
        st.metric("Forensic Score", f"{forensic_data['forensic_score']*100:.1f}%")
    st.metric("Combined Threat Score", f"{combined*100:.1f}%")

    # Signal bars
    st.markdown('<div class="section-header">Signal Breakdown</div>', unsafe_allow_html=True)
    st.markdown(score_bar_html("CNN Confidence", cnn_score, "#ff4b6e" if prediction=="FAKE" else "#00e5a0"), unsafe_allow_html=True)
    if forensic_data:
        st.markdown(score_bar_html("ELA Score",   forensic_data["ela_score"],   "#a855f7"), unsafe_allow_html=True)
        st.markdown(score_bar_html("Noise Inconsistency", forensic_data["noise_score"],  "#3b82f6"), unsafe_allow_html=True)
        st.markdown(score_bar_html("Metadata Suspicion",  forensic_data["metadata"]["suspicion"], "#f59e0b"), unsafe_allow_html=True)

    st.image(img_rgb, caption="Original Image", use_container_width=True)

# ─── Middle: Grad-CAM + ELA ──────────────────────────────────────────────────
with col_mid:
    st.markdown('<div class="section-header">Grad-CAM Attention Map</div>', unsafe_allow_html=True)
    st.image(overlay, caption="Red regions = high-attention areas (CNN)", use_container_width=True)
    st.caption("Highlights spatial regions most influential to the CNN's decision.")

    if forensic_data:
        st.markdown('<div class="section-header">Error Level Analysis (ELA)</div>', unsafe_allow_html=True)
        st.image(forensic_data["ela_img"], caption=f"ELA – Score: {forensic_data['ela_score']*100:.1f}%", use_container_width=True)
        st.caption("Bright patches reveal double-compressed or pasted regions.")

# ─── Right: Noise + FFT + Metadata ───────────────────────────────────────────
with col_right:
    if forensic_data:
        st.markdown('<div class="section-header">Noise Pattern Analysis</div>', unsafe_allow_html=True)
        st.image(forensic_data["noise_img"],
                 caption=f"Noise Map – Inconsistency: {forensic_data['noise_score']*100:.1f}%",
                 use_container_width=True)
        st.caption("Inconsistent noise patterns indicate tampering or synthesis.")

    if fft_img is not None:
        st.markdown('<div class="section-header">FFT Frequency Spectrum</div>', unsafe_allow_html=True)
        st.image(fft_img, caption="Frequency Domain – GAN Fingerprints", use_container_width=True)
        st.caption("GAN-generated images often show periodic patterns in the frequency spectrum.")

    if forensic_data:
        st.markdown('<div class="section-header">Metadata Analysis</div>', unsafe_allow_html=True)
        meta = forensic_data["metadata"]
        st.markdown(f"**EXIF present:** {'Yes' if meta['has_exif'] else 'No'}")
        if meta["software"]:
            st.markdown(f"**Software:** `{meta['software']}`")
        for flag in meta["flags"]:
            icon = "flag-item" if "⚠" in flag else "flag-ok"
            st.markdown(f'<div class="{icon}">{flag}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Forensic Report Download
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown('<div class="section-header">Download Forensic Report</div>', unsafe_allow_html=True)

report_lines = [
    "=" * 60,
    "   DEEPFAKE FORENSIC INVESTIGATION REPORT",
    "=" * 60,
    f"Date / Time : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"File        : {uploaded_file.name}",
    f"Image Size  : {pil_img.size[0]} x {pil_img.size[1]} px",
    "",
    "── CNN Analysis ──────────────────────────────────────",
    f"CNN Prediction      : {prediction}",
    f"Fake Probability    : {prob*100:.4f}%",
    f"Detection Threshold : {threshold}",
]
if forensic_data:
    meta = forensic_data["metadata"]
    report_lines += [
        "",
        "── Forensic Signal Scores ───────────────────────────",
        f"ELA Score           : {forensic_data['ela_score']*100:.2f}%",
        f"Noise Inconsistency : {forensic_data['noise_score']*100:.2f}%",
        f"Metadata Suspicion  : {meta['suspicion']*100:.2f}%",
        f"Combined Forensic   : {forensic_data['forensic_score']*100:.2f}%",
        "",
        "── Metadata Flags ───────────────────────────────────",
    ]
    for flag in meta["flags"]:
        report_lines.append(f"  • {flag}")
report_lines += [
    "",
    "── Final Verdict ─────────────────────────────────────",
    f"Combined Threat Score : {combined*100:.2f}%",
    f"FINAL VERDICT         : {final_verdict}",
    "",
    "=" * 60,
    "Generated by DeepFake Forensic Suite (EfficientNet-B0 + Forensics)",
    "=" * 60,
]
report_text = "\n".join(report_lines)

st.download_button(
    label="📄 Download Forensic Report (.txt)",
    data=report_text,
    file_name=f"forensic_report_{uploaded_file.name}.txt",
    mime="text/plain",
)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "DeepFake Forensic Suite | EfficientNet-B0 + Frequency Branch + ELA + Noise Analysis | "
    "GNA University Minor Project"
)