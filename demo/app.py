"""
Arabic TTS Demo - Audio Sample Player
Plays pre-generated audio samples comparing baseline vs fine-tuned output.
Lightweight - no model loading required, works on free CPU tier.
"""

import gradio as gr
from huggingface_hub import hf_hub_download
import os
import tempfile

MODEL_REPO = "Moeeldouma/arabic-tts-xtts-v2"

SAMPLES = {
    "Baseline (Original XTTS-v2)": "samples/Original_Model_test.wav",
    "Fine-tuned (Egyptian Arabic)": "samples/Finetuned_Model_test.wav",
    "Fine-tuned + 48kHz Upsampled": "samples/Finetuned_Model_test_48kHz.wav",
}

DEMO_TEXT = """Standard Demo Text:

مرحباً بكم في عالم التكنولوجيا الحديثة.
تُعدّ اللغة العربية من أقدم اللغات وأكثرها انتشاراً في العالم،
إذ يتحدث بها أكثر من أربعمائة مليون شخص حول العالم."""

# Download samples on startup
print("Downloading audio samples...")
sample_dir = os.path.join(tempfile.gettempdir(), "arabic-tts-samples")
os.makedirs(sample_dir, exist_ok=True)

local_paths = {}
for name, remote_path in SAMPLES.items():
    filename = os.path.basename(remote_path)
    local = os.path.join(sample_dir, filename)
    if not os.path.exists(local):
        hf_hub_download(repo_id=MODEL_REPO, filename=remote_path, local_dir=sample_dir)
        downloaded = os.path.join(sample_dir, remote_path)
        if os.path.exists(downloaded) and downloaded != local:
            os.rename(downloaded, local)
    local_paths[name] = local
    print(f"  Ready: {name}")

print("All samples loaded!")


def play_sample(choice):
    path = local_paths.get(choice)
    if path and os.path.exists(path):
        return path
    return None


with gr.Blocks(title="Arabic TTS Demo") as demo:
    gr.Markdown(
        """
        # Arabic TTS - Fine-tuned XTTS-v2

        Listen to the progression from base model to fine-tuned Arabic speech.
        All samples use the same Arabic text for direct comparison.

        **Model:** [Moeeldouma/arabic-tts-xtts-v2](https://huggingface.co/Moeeldouma/arabic-tts-xtts-v2)
        | **Code:** [GitHub](https://github.com/eldoumamoe-glitch/arabic-tts)
        | **Author:** Moe Eldouma
        """
    )

    gr.Markdown(DEMO_TEXT)

    with gr.Row():
        selector = gr.Radio(
            choices=list(SAMPLES.keys()),
            value="Fine-tuned (Egyptian Arabic)",
            label="Select Version",
        )

    audio_out = gr.Audio(label="Generated Arabic Speech", type="filepath")

    selector.change(fn=play_sample, inputs=selector, outputs=audio_out)

    # Auto-play on load
    demo.load(fn=lambda: play_sample("Fine-tuned (Egyptian Arabic)"), outputs=audio_out)

    gr.Markdown(
        """
        ---
        **How to use the full model locally:**
        ```python
        pip install TTS transformers==4.44.2
        # See GitHub repo for complete instructions
        ```

        **Limitations:** This demo plays pre-generated samples. For custom text,
        download the model and run locally with GPU. See the
        [GitHub repository](https://github.com/eldoumamoe-glitch/arabic-tts) for instructions.

        Future versions will support Sudanese, Gulf, Levantine, and MSA dialects.
        """
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
