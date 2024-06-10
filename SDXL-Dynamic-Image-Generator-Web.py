import gradio as gr
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

class ImageGeneratorApp:
    def __init__(self):
        self.pipe = None
        self.previous_frame = None
        self.load_model()

    def load_model(self):
        self.pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        self.pipe.to("cuda")

    def generate_image(self, prompt, strength, guidance_scale, num_inference_steps, seed):
        torch.manual_seed(seed)

        if self.previous_frame is None:
            init_image = Image.new('RGB', (512, 512), color='white')
            transformed_image = self.pipe(prompt=prompt,
                                          image=init_image,
                                          strength=strength,
                                          guidance_scale=guidance_scale,
                                          num_inference_steps=num_inference_steps).images[0]
        else:
            transformed_image = self.pipe(prompt=prompt,
                                          image=self.previous_frame,
                                          strength=strength,
                                          guidance_scale=guidance_scale,
                                          num_inference_steps=num_inference_steps).images[0]

        self.previous_frame = transformed_image
        return transformed_image

    def launch(self):
        with gr.Blocks() as demo:
            gr.Markdown("# Image Generator App")

            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(label="Prompt", value="A photograph of a duck", lines=2)
                    strength_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.1, label="Strength")
                    guidance_scale_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.1, label="Guidance Scale")
                    num_steps_slider = gr.Slider(minimum=1, maximum=50, value=2, step=1, label="Num Inference Steps")
                    seed_slider = gr.Slider(minimum=0, maximum=10000, value=42, step=1, label="Seed")

                with gr.Column():
                    output_image = gr.Image(type="pil", label="Generated Image")

            inputs = [text_input, strength_slider, guidance_scale_slider, num_steps_slider, seed_slider]
            
            # Initial load function
            def initial_load():
                return self.generate_image(
                    text_input.value,
                    strength_slider.value,
                    guidance_scale_slider.value,
                    num_steps_slider.value,
                    seed_slider.value,
                )

            demo.load(fn=initial_load, inputs=[], outputs=output_image)

            for input_component in inputs:
                input_component.change(fn=self.generate_image, inputs=inputs, outputs=output_image)

        demo.launch(server_name="0.0.0.0")

if __name__ == '__main__':
    app = ImageGeneratorApp()
    app.launch()
