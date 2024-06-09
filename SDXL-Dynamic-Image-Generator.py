import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageEnhance
import torch
from diffusers import AutoPipelineForImage2Image
import numpy as np
import random

class ImageGeneratorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry('1200x1350')  # Adjusted window size to accommodate larger canvas and controls

        self.output_canvas = tk.Canvas(window, width=1024, height=1024)
        self.output_canvas.grid(row=0, column=0, padx=10, pady=2)

        self.controls_frame = ttk.Frame(window)
        self.controls_frame.grid(row=1, column=0, pady=2, sticky="ew")

        self.setup_ui()

        self.recording = False
        self.previous_frame = None  # To store the previous frame for img2img feedback
        self.frame_count = 0

        self.load_model()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        font_size = ('Helvetica', 16)
        style = ttk.Style()
        style.configure('W.TButton', font=('Helvetica', 16))

        # Default text prompt
        default_prompt = "A photograph of a water drop"

        # Text input prompt
        self.text_input_label = tk.Label(self.controls_frame, text="Prompt:", font=font_size)
        self.text_input_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.text_input = tk.Text(self.controls_frame, width=40, height=2, font=font_size, wrap=tk.WORD)
        self.text_input.insert(tk.END, default_prompt)  # Set default prompt
        self.text_input.grid(row=0, column=1, padx=10, pady=10, columnspan=4, sticky="ew")

        # Adjusted slider ranges and defaults
        self.strength_slider = tk.Scale(self.controls_frame, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, label="Strength", length=200)
        self.strength_slider.set(1.0)  # Set default
        self.strength_slider.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.guidance_scale_slider = tk.Scale(self.controls_frame, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, label="Guidance Scale", length=200)
        self.guidance_scale_slider.set(1.0)  # Set default
        self.guidance_scale_slider.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.num_steps_slider = tk.Scale(self.controls_frame, from_=1, to=50, resolution=1, orient=tk.HORIZONTAL, label="Num Inference Steps", length=200)
        self.num_steps_slider.set(2)  # Set default
        self.num_steps_slider.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

        # New seed slider
        self.seed_slider = tk.Scale(self.controls_frame, from_=0, to=10000, resolution=1, orient=tk.HORIZONTAL, label="Seed", length=200)
        self.seed_slider.set(1)  # Set default to 1
        self.seed_slider.grid(row=1, column=3, padx=5, pady=5, sticky="ew")

        self.btn_toggle_record = ttk.Button(self.controls_frame, text="Toggle Generation", command=self.toggle_recording, width=20, style='W.TButton')
        self.btn_toggle_record.grid(row=2, column=0, padx=10, pady=10, columnspan=4)

    def load_model(self):
        self.pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        self.pipe.to("cuda")

    def toggle_recording(self):
        # Toggles between recording and not recording states
        self.recording = not self.recording
        if self.recording:
            print("Generation started...")
            self.generate_images()
        else:
            print("Generation stopped.")

    def generate_images(self):
        if self.recording:
            self.process_and_display_frame()
            self.window.after(10, self.generate_images)  # Continue generating images every x ms

    def process_and_display_frame(self):
        prompt = self.text_input.get("1.0", tk.END).strip()
        seed = self.seed_slider.get()

        if prompt:
            torch.manual_seed(seed)

            # Create an initial image from the prompt if it's the first frame
            if self.previous_frame is None:
                init_image = Image.new('RGB', (512, 512), color='white')
                transformed_image = self.pipe(prompt=prompt,
                                              image=init_image,
                                              strength=self.strength_slider.get(),
                                              guidance_scale=self.guidance_scale_slider.get(),
                                              num_inference_steps=self.num_steps_slider.get()).images[0]
                self.previous_frame = transformed_image
            else:
                # Apply random perturbations to the previous frame
                perturbed_image = self.apply_random_perturbations(self.previous_frame)

                # Use the previous frame for the next iteration
                transformed_image = self.pipe(prompt=prompt,
                                              image=perturbed_image,
                                              strength=self.strength_slider.get(),
                                              guidance_scale=self.guidance_scale_slider.get(),
                                              num_inference_steps=self.num_steps_slider.get()).images[0]

                # Blend the previous and current frames with a reduced effect
                blended_image = self.blend_images(self.previous_frame, transformed_image, alpha=0.1)

                self.previous_frame = blended_image  # Update the previous frame for the next iteration

                self.display_transformed_image(blended_image)

                # Increment the seed every few frames
                self.frame_count += 1
                if self.frame_count % 20 == 0:  # Change the seed every 20 frames
                    self.seed_slider.set(seed + 1)

    def apply_random_perturbations(self, image):
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1 + random.uniform(-0.05, 0.05))  # Adjust brightness slightly
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1 + random.uniform(-0.05, 0.05))  # Adjust contrast slightly
        return image

    def blend_images(self, prev_img, curr_img, alpha=0.1):
        prev_array = np.array(prev_img)
        curr_array = np.array(curr_img)
        blended_array = (alpha * prev_array + (1 - alpha) * curr_array).astype(np.uint8)
        return Image.fromarray(blended_array)

    def display_transformed_image(self, transformed_image):
        photo = ImageTk.PhotoImage(transformed_image.resize((1024, 1024), Image.LANCZOS))
        self.output_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.output_canvas.image = photo  # Keep a reference!

    def on_closing(self):
        # Properly closes the application and releases resources
        self.recording = False
        self.window.destroy()

def main():
    root = tk.Tk()
    app = ImageGeneratorApp(root, "Image Generator App")
    root.mainloop()

if __name__ == '__main__':
    main()
