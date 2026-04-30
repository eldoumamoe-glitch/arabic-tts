"""
Arabic TTS Demo — Live inference on HuggingFace Spaces (ZeroGPU).
Users can type custom Arabic text or listen to pre-generated samples.
"""

import os
import re
import tempfile
import time
import unicodedata

import gradio as gr
import numpy as np
import soundfile as sf
import spaces
import torch
from huggingface_hub import hf_hub_download
from num2words import num2words
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_REPO = "Moeeldouma/arabic-tts-xtts-v2"
SAMPLE_RATE = 24000
MAX_CHARS = 500  # safety cap for the demo
MAX_CHARS_PER_CHUNK = 160

# Generation defaults (prosody-tuned for Arabic)
DEFAULT_TEMP = 0.55
DEFAULT_TOP_P = 0.85
DEFAULT_REP_PENALTY = 2.5

# Pre-generated samples hosted on the model repo
SAMPLES = {
    "BasicModel(Arabic)": "samples/base_natural.wav",
    "FinetunedModel(Arabic)": "samples/finetuned_natural.wav",
}

DEMO_TEXT = "مرحباً بكم في عالم التكنولوجيا الحديثة. تُعدّ اللغة العربية من أقدم اللغات وأكثرها انتشاراً في العالم، إذ يتحدث بها أكثر من أربعمائة مليون شخص حول العالم."

# ---------------------------------------------------------------------------
# Arabic Preprocessor (inline to keep the Space self-contained)
# ---------------------------------------------------------------------------
HAMZA_CORRECTIONS = {
    "ان": "أن", "انا": "أنا", "انت": "أنت", "انتم": "أنتم",
    "اكثر": "أكثر", "اقل": "أقل", "اول": "أول", "اي": "أي",
    "ايضا": "أيضاً", "اذا": "إذا", "امام": "أمام", "اصبح": "أصبح",
    "اصبحت": "أصبحت", "اخرى": "أخرى", "اخر": "آخر", "اكبر": "أكبر",
    "اهم": "أهم", "الى": "إلى", "اذ": "إذ", "انما": "إنما",
    "انه": "إنه", "انها": "إنها", "الان": "الآن", "الالات": "الآلات",
    "مسوول": "مسؤول", "روية": "رؤية", "تاثير": "تأثير", "سوال": "سؤال",
}

SYMBOL_MAP = {"&": " و ", "%": " بالمئة", "$": " دولار", "°": " درجة"}

_hamza_pattern = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in HAMZA_CORRECTIONS) + r")\b"
)
SENTENCE_ENDINGS = re.compile(r"[.!?؟。！？]+")
SECONDARY_SPLITS = re.compile(r"[،؛,;]+")


def preprocess_arabic(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]", "", text)
    text = _hamza_pattern.sub(lambda m: HAMZA_CORRECTIONS.get(m.group(0), m.group(0)), text)
    # Numbers
    def _num(m):
        try:
            n = float(m.group(0)) if "." in m.group(0) else int(m.group(0))
            return num2words(n, lang="ar")
        except (ValueError, OverflowError):
            return m.group(0)
    def _pct(m):
        try:
            n = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
            return num2words(n, lang="ar") + " بالمئة"
        except (ValueError, OverflowError):
            return m.group(0)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*[%٪]", _pct, text)
    text = re.sub(r"\d+(?:\.\d+)?", _num, text)
    for sym, rep in SYMBOL_MAP.items():
        text = text.replace(sym, rep)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text, max_chars=MAX_CHARS_PER_CHUNK):
    sentences = SENTENCE_ENDINGS.split(text)
    delimiters = SENTENCE_ENDINGS.findall(text)
    parts = []
    for i, s in enumerate(sentences):
        s = s.strip()
        if not s:
            continue
        if i < len(delimiters):
            s += delimiters[i]
        parts.append(s)

    chunks = []
    for sent in parts:
        if len(sent) <= max_chars:
            chunks.append(sent)
        else:
            # split at commas
            sub = SECONDARY_SPLITS.split(sent)
            cur = ""
            for p in sub:
                p = p.strip()
                if not p:
                    continue
                if cur and len(cur) + len(p) + 2 > max_chars:
                    chunks.append(cur)
                    cur = p
                else:
                    cur = (cur + "، " + p).strip("، ") if cur else p
            if cur:
                chunks.append(cur)

    # merge tiny chunks
    if len(chunks) > 1:
        merged = [chunks[0]]
        for c in chunks[1:]:
            if len(merged[-1]) < 30 or len(c) < 30:
                merged[-1] += " " + c
            else:
                merged.append(c)
        chunks = merged
    return chunks


# ---------------------------------------------------------------------------
# Model loading (runs once on Space startup — CPU)
# ---------------------------------------------------------------------------
print("⏳ Downloading model files...")
cache_dir = os.path.join(tempfile.gettempdir(), "arabic-tts-model")
os.makedirs(cache_dir, exist_ok=True)

# Download base XTTS-v2 via TTS library
from TTS.utils.manage import ModelManager
mm = ModelManager()
model_path, config_path, _ = mm.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
base_dir = os.path.dirname(config_path)

# Download finetuned checkpoint from our HF repo
ft_checkpoint = hf_hub_download(repo_id=MODEL_REPO, filename="best_model.pth")

# Download sample files
sample_dir = os.path.join(tempfile.gettempdir(), "arabic-tts-samples")
os.makedirs(sample_dir, exist_ok=True)
local_samples = {}
for name, remote in SAMPLES.items():
    local = os.path.join(sample_dir, os.path.basename(remote))
    if not os.path.exists(local):
        downloaded = hf_hub_download(repo_id=MODEL_REPO, filename=remote, local_dir=sample_dir)
        if os.path.exists(os.path.join(sample_dir, remote)) and not os.path.exists(local):
            os.rename(os.path.join(sample_dir, remote), local)
    local_samples[name] = local

# Load model on CPU
print("⏳ Loading XTTS-v2 base model...")
config = XttsConfig()
config.load_json(os.path.join(base_dir, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=base_dir)

# Overlay finetuned GPT weights
print("⏳ Loading finetuned weights...")
ckpt = torch.load(ft_checkpoint, map_location="cpu", weights_only=False)
model_state = ckpt.get("model", ckpt)
stripped = {}
for k, v in model_state.items():
    stripped[k.replace("xtts.", "", 1) if k.startswith("xtts.") else k] = v
current = model.state_dict()
matched = {k: v for k, v in stripped.items() if k in current and v.shape == current[k].shape}
current.update(matched)
model.load_state_dict(current)
print(f"✅ Loaded {len(matched)}/{len(stripped)} finetuned weights")
model.eval()

# Pre-compute speaker conditioning from a built-in speaker (no ref audio in Space)
# Use a built-in speaker for conditioning since we can't ship training wavs
print("⏳ Computing speaker conditioning...")
speakers = torch.load(os.path.join(base_dir, "speakers_xtts.pth"), weights_only=False)
speaker_data = speakers["Gilberto Mathias"]
gpt_cond_latent = speaker_data["gpt_cond_latent"]
speaker_embedding = speaker_data["speaker_embedding"]
print("✅ Model ready!")


# ---------------------------------------------------------------------------
# Inference (runs on ZeroGPU)
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def generate_speech(text, temperature, top_p, rep_penalty):
    if not text or not text.strip():
        raise gr.Error("Please enter some Arabic text.")

    text = text.strip()
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Text too long ({len(text)} chars). Maximum is {MAX_CHARS} characters.")

    # Move model to GPU
    model.cuda()
    gcl = gpt_cond_latent.cuda()
    se = speaker_embedding.cuda()

    # Preprocess and chunk
    processed = preprocess_arabic(text)
    chunks = chunk_text(processed)

    pause = np.zeros(int(SAMPLE_RATE * 0.4))
    all_wav = []
    t0 = time.time()

    for i, chunk in enumerate(chunks):
        torch.manual_seed(12345 + i)
        torch.cuda.manual_seed(12345 + i)

        out = model.inference(
            text=chunk,
            language="ar",
            gpt_cond_latent=gcl,
            speaker_embedding=se,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=rep_penalty,
        )
        all_wav.append(out["wav"])
        if i < len(chunks) - 1:
            all_wav.append(pause)

    gen_time = time.time() - t0
    final_wav = np.concatenate(all_wav)
    duration = len(final_wav) / SAMPLE_RATE

    # Save to temp file
    out_path = os.path.join(tempfile.gettempdir(), "arabic_tts_output.wav")
    sf.write(out_path, final_wav, SAMPLE_RATE)

    info = f"Duration: {duration:.1f}s | Generated in {gen_time:.1f}s | RTF: {gen_time/duration:.2f} | Chunks: {len(chunks)}"
    return out_path, info


def play_sample(choice):
    path = local_samples.get(choice)
    if path and os.path.exists(path):
        return path, f"Pre-generated sample: {choice}"
    return None, ""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Arabic TTS - XTTS-v2", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Arabic TTS - Fine-tuned XTTS-v2

        Type Arabic text and generate speech, or listen to pre-generated samples.

        **Model:** [Moeeldouma/arabic-tts-xtts-v2](https://huggingface.co/Moeeldouma/arabic-tts-xtts-v2)
        | **Author:** Moe Eldouma
        """
    )

    with gr.Tabs():
        # --- Tab 1: Live Inference ---
        with gr.TabItem("Generate Speech"):
            text_input = gr.Textbox(
                label="Arabic Text",
                placeholder="اكتب نصاً عربياً هنا...",
                value=DEMO_TEXT,
                lines=4,
                max_lines=8,
                rtl=True,
            )

            with gr.Row():
                temperature = gr.Slider(0.1, 1.0, value=DEFAULT_TEMP, step=0.05, label="Temperature")
                top_p = gr.Slider(0.5, 1.0, value=DEFAULT_TOP_P, step=0.05, label="Top-p")
                rep_penalty = gr.Slider(1.0, 10.0, value=DEFAULT_REP_PENALTY, step=0.5, label="Repetition Penalty")

            generate_btn = gr.Button("Generate", variant="primary", size="lg")

            audio_output = gr.Audio(label="Generated Speech", type="filepath")
            info_output = gr.Textbox(label="Info", interactive=False)

            generate_btn.click(
                fn=generate_speech,
                inputs=[text_input, temperature, top_p, rep_penalty],
                outputs=[audio_output, info_output],
            )

        # --- Tab 2: Pre-generated Samples ---
        with gr.TabItem("Sample Comparison"):
            gr.Markdown(
                f"""**Demo text:** {DEMO_TEXT}"""
            )
            selector = gr.Radio(
                choices=list(SAMPLES.keys()),
                value="FinetunedModel(Arabic)",
                label="Select Version",
            )
            sample_audio = gr.Audio(label="Sample Audio", type="filepath")
            sample_info = gr.Textbox(label="Info", interactive=False)

            selector.change(fn=play_sample, inputs=selector, outputs=[sample_audio, sample_info])
            demo.load(fn=lambda: play_sample("FinetunedModel(Arabic)"), outputs=[sample_audio, sample_info])

    gr.Markdown(
        """
        ---
        **Note:** This demo uses ZeroGPU for free GPU inference. Generation may take 10-30 seconds
        depending on text length. Maximum 500 characters per request.

        **Limitations:** Currently trained on Egyptian Arabic. Future versions will support
        more dialects and emotions. Contributions welcome!
        """
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
