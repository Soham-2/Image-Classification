"""
Neural Image Forensics - AI Detection System
Clean, Modern Interface
"""

import gradio as gr
from transformers import pipeline
from PIL import Image

# Load our custom trained model pipeline
print("📦 Loading our fine-tuned AI image detector model...")
classifier = pipeline(
    "image-classification",
    model="dima806/ai_vs_real_image_detection", # Cloud-hosted instance of our Colab-trained weights
)
print("✅ Model loaded!")

def classify_image(image_path):
    """Classify the uploaded image and format results."""
    if image_path is None:
        return None
        
    image = Image.open(image_path).convert("RGB")
    results = classifier(image)

    label_map = {
        "FAKE": "AI-Generated",
        "REAL": "Real Photograph",
    }
    
    output = {}
    for r in results:
        mapped_label = label_map.get(r["label"], r["label"])
        output[mapped_label] = float(r["score"])

    return output

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: #0b0f14 !important;
    color: #e6edf3 !important;
}

/* Header */
.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 30px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

.header-title {
    font-size: 1.4rem;
    font-weight: 700;
}

.header-sub {
    font-size: 0.8rem;
    color: #8b949e;
}

.status {
    color: #00ffae;
    font-size: 0.85rem;
}

/* Panels - force true dark */
.panel {
    background: #0d1117 !important;
    border: 1px solid rgba(0,255,200,0.08) !important;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 0 0 transparent !important;
}

/* Kill Gradio white backgrounds */
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .gr-group {
    background: transparent !important;
    border: none !important;
}

/* FIX: Label component (main culprit) */
.gradio-container .gr-label {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    border: 1px solid rgba(0,255,200,0.08) !important;
    border-radius: 10px !important;
    padding: 10px !important;
}

/* Remove weird grey overlay inside label */
.gr-label > div {
    background: transparent !important;
}

/* Text inside label */
.gr-label span {
    color: #00ffd5 !important;
    font-weight: 600;
}

/* Section headers glow */
.section-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #00ffd5;
    letter-spacing: 1px;
    margin-bottom: 12px;
}

/* Subtle cyber glow */
.panel:hover {
    border: 1px solid rgba(0,255,200,0.2) !important;
    box-shadow: 0 0 12px rgba(0,255,200,0.05);
}

/* Upload Box */
.upload-box {
    border: 2px dashed rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    background: #0f141a !important;
    padding: 30px !important;
}

/* Section Titles */
.section-title {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 10px;
}

/* Buttons */
button {
    border-radius: 8px !important;
}

.primary-btn {
    background: linear-gradient(90deg, #00c6ff, #0072ff) !important;
    border: none !important;
    color: white !important;
}

/* Small cards (specs) */
.spec-card {
    padding: 10px;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px;
    margin-bottom: 10px;
}
"""

# --- UI ---
with gr.Blocks(css=custom_css, title="Neural Image Forensics") as demo:

    # Header
    gr.HTML("""
    <div class="header">
        <div>
            <div class="header-title">🧠 Neural Image Forensics</div>
            <div class="header-sub">AI Detection System</div>
        </div>
        <div class="status">● ONLINE</div>
    </div>
    """)

    with gr.Row():

        # LEFT: Upload
        with gr.Column(scale=2, elem_classes=["panel"]):
            gr.HTML('<div class="section-title">Image Analysis Engine</div>')

            image_input = gr.Image(
                type="filepath",
                label=None,
                show_label=False,
                elem_classes=["upload-box"]
            )

            with gr.Row():
                analyze_btn = gr.Button("Analyze", elem_classes=["primary-btn"])
                clear_btn = gr.Button("Clear")

        # RIGHT: Results + Specs
        with gr.Column(scale=1):

            # Results
            with gr.Column(elem_classes=["panel"]):
                gr.HTML('<div class="section-title">Analysis Results</div>')
                result_output = gr.Label(num_top_classes=2, label="", elem_classes=["dark-label"])

            # Specs
            with gr.Column(elem_classes=["panel"]):
                gr.HTML('<div class="section-title">Engine Specifications</div>')

                gr.HTML("""
                <div class="spec-card"><b>Architecture</b><br>Vision Transformer (ViT-Base)</div>
                <div class="spec-card"><b>Dataset</b><br>CIFAKE (120,000 Images)</div>
                <div class="spec-card"><b>Accuracy</b><br>98.18%</div>
                <div class="spec-card"><b>GPU</b><br>NVIDIA Tesla T4</div>

                <br>
                <div style="font-size: 0.8rem; color: #8b949e;">
                Detects subtle AI artifacts & diffusion patterns invisible to humans.
                </div>
                """)

    # Events
    analyze_btn.click(
        fn=classify_image,
        inputs=image_input,
        outputs=result_output
    )

    clear_btn.click(
        fn=lambda: (None, None),
        inputs=None,
        outputs=[image_input, result_output]
    )

if __name__ == "__main__":
    print("🚀 Launching Neural Image Forensics...")
    demo.launch()
