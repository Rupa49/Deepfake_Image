"""
app.py  –  DeepFake Forensic Investigation Suite  v3.0
"""
import io, datetime
import numpy as np
import cv2
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
# pyrefly: ignore [missing-import]
from model     import DeepfakeDetector
from forensics import run_full_forensics
import plotly.graph_objects as go

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="DeepFake Forensic Suite", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] { background:#0a0c10; color:#e2e8f0; }
[data-testid="stSidebar"]          { background:#0f131a; border-right:1px solid #1e2535; }

/* Futuristic Header */
.main-title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
    padding-bottom: 0px;
    letter-spacing: -1px;
}
.sub-title {
    text-align: center;
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: -10px;
    margin-bottom: 30px;
}

.verdict-fake { font-size:3rem; font-weight:800; color:#ff4b6e; text-align: center;
                text-shadow:0 0 30px rgba(255,75,110,0.6); letter-spacing:.08em; 
                padding: 10px; border: 1px solid rgba(255,75,110,0.3); border-radius: 12px; background: rgba(255,75,110,0.05);}
.verdict-real { font-size:3rem; font-weight:800; color:#00e5a0; text-align: center;
                text-shadow:0 0 30px rgba(0,229,160,0.6); letter-spacing:.08em; 
                padding: 10px; border: 1px solid rgba(0,229,160,0.3); border-radius: 12px; background: rgba(0,229,160,0.05);}
.section-header { font-size:.85rem; font-weight:700; letter-spacing:.15em;
                  color:#00C9FF; text-transform:uppercase; margin:24px 0 12px 0;
                  border-left:4px solid #00C9FF; padding-left:10px; }
.why-box  { background: rgba(109, 40, 217, 0.1); border:1px solid rgba(109, 40, 217, 0.5); border-radius:12px;
            padding:16px; margin:12px 0; font-size:.9rem; color:#e9d5ff; backdrop-filter: blur(5px); }
.flag-item{ font-size:.85rem; color:#fbbf24; padding:4px 0; }
.flag-ok  { font-size:.85rem; color:#34d399; padding:4px 0; }
.override-badge { background: linear-gradient(90deg, #6d28d9, #9333ea); color:#fff; font-size:.8rem;
                  font-weight:700; padding:6px 14px; border-radius:20px;
                  letter-spacing:.1em; display:block; text-align:center; margin-top:12px; margin-bottom: 12px; box-shadow: 0 0 15px rgba(109, 40, 217, 0.4);}
div[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.03); border:1px solid rgba(255, 255, 255, 0.1); 
    border-radius:12px; padding:15px; box-shadow: 0 4px 20px rgba(0,0,0,0.2); 
    transition: transform 0.3s ease, box-shadow 0.3s ease; }
div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0, 201, 255, 0.15);
    border: 1px solid rgba(0, 201, 255, 0.3);
}

@keyframes loadBar {
  0% { width: 0; }
}
.animated-bar {
  animation: loadBar 1.5s cubic-bezier(0.25, 1, 0.5, 1) forwards;
}
</style>""", unsafe_allow_html=True)


# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DeepfakeDetector().to(device)
    try:
        model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
        model.eval()
        status = "✅ Trained weights loaded"
    except FileNotFoundError:
        model.eval()
        status = "⚠️ Demo mode – no trained weights found"
    return model, device, status

model, device, model_status = load_model()


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────
def grad_cam(model, input_tensor, img_rgb):
    target_layer       = model.gradcam_target
    activations, grads = {}, {}
    h1 = target_layer.register_forward_hook(
        lambda m, i, o: activations.update({"f": o.detach()}))
    h2 = target_layer.register_full_backward_hook(
        lambda m, i, o: grads.update({"b": o[0].detach()}))
    output = model(input_tensor)
    model.zero_grad()
    output.backward()
    h1.remove()
    h2.remove()
    weights = torch.mean(grads["b"], dim=(2, 3), keepdim=True)
    cam     = torch.sum(weights * activations["f"], dim=1).squeeze().cpu().numpy()
    cam     = np.maximum(cam, 0)
    cam     = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
    cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    hmap    = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.55, hmap, 0.45, 0)
    return overlay, output


# ─── FFT Spectrum ─────────────────────────────────────────────────────────────
def compute_fft(pil_img):
    gray = np.array(pil_img.convert("L")).astype(np.float32)
    mag  = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)
    norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    sp   = cv2.applyColorMap(np.uint8(norm * 255), cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(sp, cv2.COLOR_BGR2RGB)


# ─── Score Bar HTML ───────────────────────────────────────────────────────────
def score_bar(label, val, color="#4a90d9", note=""):
    pct = int(val * 100)
    bar_color = color
    return f"""
    <div style="margin:8px 0; padding: 4px;">
      <div style="display:flex;justify-content:space-between;font-size:.85rem;
                  color:#cbd5e1;margin-bottom:4px; font-weight: 600;">
        <span>{label} <span style="color:#64748b;font-size:.75rem; font-weight:400;">{note}</span></span>
        <span style="font-weight:700;color:{'#ff4b6e' if pct>60 else '#94a3b8'}">{pct}%</span>
      </div>
      <div style="background:#1e2535;border-radius:6px;height:8px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);">
        <div class="animated-bar" style="background:{bar_color};width:{pct}%;height:8px;border-radius:6px; box-shadow: 0 0 10px {bar_color};"></div>
      </div>
    </div>"""


# ─── Smart Ensemble Decision ──────────────────────────────────────────────────
def smart_verdict(cnn_prob, forensic_score, threshold, fd):
    triggered = []

    if fd["metadata"]["suspicion"] > 0.4:
        triggered.append(f"No EXIF / AI software detected ({fd['metadata']['suspicion']*100:.0f}%)")
    if fd["dct_score"] > 0.45:
        triggered.append(f"Smooth frequency spectrum – DCT score {fd['dct_score']*100:.0f}%")
    if fd["tex_score"] > 0.45:
        triggered.append(f"Uniform texture (AI skin) – LBP score {fd['tex_score']*100:.0f}%")
    if fd["ela_score"] > 0.35:
        triggered.append(f"Compression artifact anomaly – ELA {fd['ela_score']*100:.0f}%")
    if fd["edge_score"] > 0.45:
        triggered.append(f"Uniformly sharp edges – CoV {fd['sharpness_cov']:.2f}")
    if fd["noise_score"] > 0.50:
        triggered.append(f"Inconsistent sensor noise – {fd['noise_score']*100:.0f}%")

    # Case 1: CNN very confident FAKE
    if cnn_prob > 0.70:
        combined = 0.65 * cnn_prob + 0.35 * forensic_score
        return "FAKE", combined, "CNN strongly detected face-swap manipulation.", triggered

    # Case 2: CNN confident REAL and forensics also low
    if cnn_prob < 0.25 and forensic_score < 0.30:
        combined = 0.65 * cnn_prob + 0.35 * forensic_score
        return "REAL", combined, "CNN and all forensic signals indicate authentic image.", triggered

    # Case 3: FORENSIC OVERRIDE — catches AI-generated images
    if forensic_score > 0.45 and cnn_prob < 0.65:
        combined = 0.35 * cnn_prob + 0.65 * forensic_score
        reason   = ("CNN did not detect face-swap, BUT forensic signals are strongly "
                    "suspicious. This pattern is typical of AI-generated images "
                    "(Midjourney / DALL-E / Stable Diffusion) which look real to a "
                    "face-swap detector but betray themselves through missing EXIF, "
                    "texture smoothness, and frequency-domain artifacts.")
        return "FAKE", min(combined, 1.0), reason, triggered

    # Case 4: Standard ensemble
    combined = 0.55 * cnn_prob + 0.45 * forensic_score
    verdict  = "FAKE" if combined > threshold else "REAL"
    return verdict, min(combined, 1.0), "Standard ensemble of CNN + forensic signals.", triggered


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Forensic Suite v3.0")
    st.caption(model_status)
    st.divider()
    st.markdown("### ⚙️ Engine Settings")
    threshold    = st.slider("Detection Threshold", 0.1, 0.9, 0.50, 0.01)
    run_forensic = st.toggle("Run Multi-Signal Forensics", value=True)
    show_fft     = st.toggle("Show FFT Spectrum", value=True)
    show_new     = st.toggle("Show DCT / Texture / Edge panels", value=True)
    st.divider()
    st.markdown("""
**Signal guide**
- 🔴 Grad-CAM → CNN attention
- 🟣 ELA → compression artifacts
- 🔵 Noise → sensor inconsistency
- 🟡 FFT → GAN fingerprints
- 🟢 DCT → frequency smoothness *(NEW)*
- 🟠 Texture → LBP uniformity *(NEW)*
- ⚪ Edge → sharpness profile *(NEW)*
""")
    st.divider()
    st.info("💡 **v3.0 Engine:** Forensic signals override CNN to catch advanced AI-generated images (Midjourney, DALL-E).")


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🛡️ DeepFake Forensic Investigation</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced 6-Signal Telemetry: CNN + ELA + Noise + DCT + Texture + Edge</p>', unsafe_allow_html=True)
st.divider()

# ─── File Uploader ────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload image for deep forensic analysis",
    type=["jpg", "jpeg", "png"]
)

# ══════════════════════════════════════════════════════════════════════════════
if uploaded is not None:

    # ── Read & decode image ───────────────────────────────────────────────────
    file_bytes   = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    img_rgb      = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    pil_img      = Image.fromarray(img_rgb)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    input_tensor = tf(pil_img).unsqueeze(0).to(device)

    # ── Run all analysis ──────────────────────────────────────────────────────
    with st.spinner("🔍 Running advanced multi-signal forensic analysis..."):

        overlay, output = grad_cam(model, input_tensor, img_rgb)
        cnn_prob        = torch.sigmoid(output).item()
        cnn_verdict     = "FAKE" if cnn_prob > threshold else "REAL"

        fd      = run_full_forensics(pil_img) if run_forensic else None
        fft_img = compute_fft(pil_img) if show_fft else None

        if fd:
            final_verdict, combined, reason, triggered = smart_verdict(
                cnn_prob, fd["forensic_score"], threshold, fd)
            forensic_override = (final_verdict == "FAKE" and cnn_verdict == "REAL")
        else:
            combined          = cnn_prob
            final_verdict     = cnn_verdict
            reason            = "Forensics disabled."
            triggered         = []
            forensic_override = False

    # ── Layout ────────────────────────────────────────────────────────────────
    col_l, col_m, col_r = st.columns([1.2, 1.1, 1.1])

    # ── LEFT column ───────────────────────────────────────────────────────────
    with col_l:
        st.markdown('<div class="section-header">Final Verdict</div>',
                    unsafe_allow_html=True)
        cls = "verdict-fake" if final_verdict == "FAKE" else "verdict-real"
        st.markdown(f'<div class="{cls}">{final_verdict}</div>',
                    unsafe_allow_html=True)



        # Plotly Radar Chart for Forensic Signature
        if fd:
            st.markdown('<div class="section-header">Forensic Signature Radar</div>', unsafe_allow_html=True)
            categories = ['Metadata Risk', 'DCT Smoothness', 'Texture Uniformity', 'ELA Artifacts', 'Noise Inconsistency', 'Edge Uniformity']
            values = [
                fd['metadata']['suspicion']*100, 
                fd['dct_score']*100, 
                fd['tex_score']*100, 
                fd['ela_score']*100, 
                fd['noise_score']*100, 
                fd['edge_score']*100
            ]
            
            fig = go.Figure()
            
            radar_color = '#ff4b6e' if final_verdict == 'FAKE' else '#00e5a0'
            fill_color = 'rgba(255,75,110,0.3)' if final_verdict == 'FAKE' else 'rgba(0,229,160,0.3)'
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name='Forensic Footprint',
                line_color=radar_color,
                fillcolor=fill_color,
                marker=dict(size=8, color=radar_color, symbol='diamond')
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], color="rgba(255,255,255,0.3)", gridcolor="rgba(255,255,255,0.1)", tickfont=dict(size=10)),
                    angularaxis=dict(color="#94a3b8", gridcolor="rgba(255,255,255,0.1)", tickfont=dict(size=11, family="Inter", color="#00C9FF"))
                ),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=30, b=30, l=40, r=40),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)


        c1, c2, c3 = st.columns(3)
        c1.metric("CNN Score", f"{cnn_prob*100:.1f}%")
        if fd:
            c2.metric("Forensic", f"{fd['forensic_score']*100:.1f}%")
        c3.metric("Threat", f"{combined*100:.1f}%")

        st.markdown('<div class="section-header">Signal Breakdown</div>',
                    unsafe_allow_html=True)
        clr = "#ff4b6e" if cnn_prob > 0.5 else "#00e5a0"
        st.markdown(score_bar("CNN (face-swap)", cnn_prob, clr, "(CelebsV2 trained)"),
                    unsafe_allow_html=True)
        if fd:
            st.markdown(score_bar("Metadata suspicion",  fd["metadata"]["suspicion"],
                                  "#f59e0b", "(no EXIF = AI-like)"), unsafe_allow_html=True)
            st.markdown(score_bar("DCT smoothness",      fd["dct_score"], "#22d3ee",
                                  f"HF={fd['hf_energy']:.3f}"), unsafe_allow_html=True)
            st.markdown(score_bar("Texture uniformity",  fd["tex_score"], "#f97316",
                                  f"LBP={fd['lbp_variance']:.2f}"), unsafe_allow_html=True)
            st.markdown(score_bar("ELA artifacts",       fd["ela_score"], "#a855f7"),
                        unsafe_allow_html=True)
            st.markdown(score_bar("Noise inconsistency", fd["noise_score"], "#3b82f6"),
                        unsafe_allow_html=True)
            st.markdown(score_bar("Edge uniformity",     fd["edge_score"], "#94a3b8",
                                  f"CoV={fd['sharpness_cov']:.2f}"), unsafe_allow_html=True)

        if triggered:
            st.markdown('<div class="section-header">Why FAKE was triggered</div>',
                        unsafe_allow_html=True)
            bullets = "".join(
                f"<div style='padding:4px 0;color:#c4b5fd'>• {t}</div>"
                for t in triggered)
            st.markdown(f'<div class="why-box">{bullets}</div>',
                        unsafe_allow_html=True)

        if forensic_override:
            st.markdown('<div class="section-header">Decision Reasoning</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="why-box" style="border-color:#f59e0b;color:#fde68a; background: rgba(245, 158, 11, 0.1);">'
                f'{reason}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Source Material</div>', unsafe_allow_html=True)
        st.image(img_rgb, caption="Original Analyzed Image", use_container_width=True)

    # ── MIDDLE column ─────────────────────────────────────────────────────────
    with col_m:
        st.markdown('<div class="section-header">Grad-CAM Attention (CNN)</div>',
                    unsafe_allow_html=True)
        st.image(overlay, caption="Red = CNN focused areas (Where the detector looked)",
                 use_container_width=True)

        if fd:
            st.markdown(
                f'<div class="section-header">Error Level Analysis (ELA)</div>',
                unsafe_allow_html=True)
            st.image(fd["ela_img"],
                     caption=f"Score: {fd['ela_score']*100:.1f}% | Bright patches = anomalies",
                     use_container_width=True)

            if show_new:
                st.markdown(
                    f'<div class="section-header">DCT Frequency Map</div>',
                    unsafe_allow_html=True)
                st.image(fd["dct_img"],
                         caption=f"Score: {fd['dct_score']*100:.1f}% | Dark = smooth (AI images lack detail)",
                         use_container_width=True)

    # ── RIGHT column ──────────────────────────────────────────────────────────
    with col_r:
        if fd:
            st.markdown(
                f'<div class="section-header">Sensor Noise Inconsistency</div>',
                unsafe_allow_html=True)
            st.image(fd["noise_img"],
                     caption=f"Score: {fd['noise_score']*100:.1f}% | Tampering leaves noise gaps",
                     use_container_width=True)

        if fft_img is not None:
            st.markdown('<div class="section-header">FFT Frequency Spectrum</div>',
                        unsafe_allow_html=True)
            st.image(fft_img,
                     caption="GAN fingerprints appear as periodic geometric patterns",
                     use_container_width=True)

        if fd and show_new:
            st.markdown(
                f'<div class="section-header">Texture Local Binary Pattern</div>',
                unsafe_allow_html=True)
            st.image(fd["tex_img"],
                     caption=f"Score: {fd['tex_score']*100:.1f}% | Low variance = AI plastic texture",
                     use_container_width=True)

            st.markdown(
                f'<div class="section-header">Edge Sharpness Map</div>',
                unsafe_allow_html=True)
            st.image(fd["edge_img"],
                     caption=f"Score: {fd['edge_score']*100:.1f}% | Uniformly sharp = AI generated",
                     use_container_width=True)

        if fd:
            st.markdown('<div class="section-header">EXIF Metadata Findings</div>',
                        unsafe_allow_html=True)
            meta = fd["metadata"]
            st.markdown(f"**EXIF present:** {'✅ Yes' if meta['has_exif'] else '❌ No (Highly Suspicious)'}")
            if meta["software"]:
                st.markdown(f"**Software footprint:** `{meta['software']}`")
            for flag in meta["flags"]:
                icon = "flag-item" if "⚠" in flag else "flag-ok"
                st.markdown(f'<div class="{icon}">{flag}</div>',
                            unsafe_allow_html=True)

    # ── Report Download ───────────────────────────────────────────────────────
    st.divider()
    report_lines = [
        "=" * 60,
        "   NEXUS DEEPFAKE FORENSIC INVESTIGATION REPORT v3.0",
        "=" * 60,
        f"Date       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"File       : {uploaded.name}",
        f"Image size : {pil_img.size[0]} x {pil_img.size[1]} px",
        "",
        "── CNN Analysis ──────────────────────────────────────────",
        f"CNN Score (face-swap) : {cnn_prob*100:.4f}%",
        f"CNN Verdict           : {cnn_verdict}",
        f"Threshold             : {threshold}",
    ]
    if fd:
        report_lines += [
            "",
            "── Forensic Signals ──────────────────────────────────────",
            f"Metadata Suspicion    : {fd['metadata']['suspicion']*100:.2f}%",
            f"DCT Smoothness Score  : {fd['dct_score']*100:.2f}%  (HF energy={fd['hf_energy']:.4f})",
            f"Texture Uniformity    : {fd['tex_score']*100:.2f}%  (LBP var={fd['lbp_variance']:.4f})",
            f"ELA Artifacts         : {fd['ela_score']*100:.2f}%",
            f"Noise Inconsistency   : {fd['noise_score']*100:.2f}%",
            f"Edge Uniformity       : {fd['edge_score']*100:.2f}%  (CoV={fd['sharpness_cov']:.4f})",
            f"Combined Forensic     : {fd['forensic_score']*100:.2f}%",
            "",
            "── Metadata Flags ────────────────────────────────────────",
        ] + [f"  • {f}" for f in fd["metadata"]["flags"]]

    if triggered:
        report_lines += ["", "── Triggered Signals ─────────────────────────────────────"]
        report_lines += [f"  ✦ {t}" for t in triggered]

    report_lines += [
        "",
        "── Final Decision ────────────────────────────────────────",
        f"Combined Threat Score : {combined*100:.2f}%",
        f"Forensic Override     : {'YES' if forensic_override else 'No'}",
        f"FINAL VERDICT         : {final_verdict}",
        f"Decision Reason       : {reason[:150]}",
        "",
        "=" * 60,
        "Generated by DeepFake Forensic Suite v3.0",
        "Advanced EfficientNet + 6-signal forensic telemetry",
        "=" * 60,
    ]

    st.download_button(
        "📄 Download Official Forensic Report (.txt)",
        data="\n".join(report_lines),
        file_name=f"forensic_report_{uploaded.name}.txt",
        mime="text/plain",
        type="primary"
    )

else:
    # ── No file uploaded yet — show instructions ──────────────────────────────
    st.info("Upload an image above to begin advanced forensic analysis.", icon="📁")
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        ### 🔍 What this engine detects
        - **Face-swap deepfakes** (Trained on CelebsV2 dataset)
        - **AI-generated images** (Midjourney, DALL-E, Stable Diffusion) 
        - **Image manipulations** (Pasted objects, clone-stamping)
        - **Compression artifacts** (Re-saved JPEG artifacts)
        """)
    with col_info2:
        st.markdown("""
        ### ⚙️ How the multi-signal telemetry works
        The system runs **6 independent forensic signals** simultaneously:
        1. CNN Face-Swap Detector
        2. Error Level Analysis (ELA)
        3. Sensor Noise Maps
        4. DCT Frequency Analysis
        5. Texture Uniformity (LBP)
        6. Edge Sharpness Profiles
        
        *It uses a smart ensemble algorithm that allows forensic evidence to **override** the CNN if AI-generation traces are found.*
        """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.9rem;'>DeepFake Forensic Suite v3.0 | GNA University Minor Project | Rupa Tiwari GU-2023-3522</p>", unsafe_allow_html=True)
