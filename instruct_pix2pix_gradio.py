import gradio as gr
from PIL import Image
import numpy as np
import torch
import os
from datetime import datetime
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Load the pipeline
print("Loading pipeline components...")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()  # Helps with 8GB VRAM

# Ensure capture folder exists
os.makedirs("captures", exist_ok=True)

# Inference function
def generate(prompt, image):
    if image is None:
        return "No image provided."

    # Convert image to PIL and resize
    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
    image_pil = image_pil.resize((512, 512), Image.LANCZOS)

    # Save captured image with timestamp for record
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"captures/captured_{timestamp}.jpg"
    image_pil.save(image_path)

    # Generate output
    result = pipe(
        prompt,
        image=image_pil,
        num_inference_steps=40,
        image_guidance_scale=1.5,
        guidance_scale=7.5,
    ).images[0]

    return result

# Gradio UI
iface = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Instruction (Prompt)"),
        gr.Image(
            label="Input Image (Upload or Capture)",
            type="numpy",
            sources=["upload", "webcam"],
            image_mode="RGB",
        ),
    ],
    outputs=gr.Image(label="Modified Image"),
    title="Visual Prompt-Based Editing",
    description="Upload or capture an image and give an edit instruction.\n"
                "Try examples like 'Make him wear a suit' or 'Make it rain'.",
    examples=[
        ["Make him wear a suit", "examples/basil.jpg"],
        ["Make it rain", "examples/DSCS.jpg"],
    ],
)

iface.launch()
