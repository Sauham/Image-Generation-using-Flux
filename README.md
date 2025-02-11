# README: Synthetic Image Generation, Preprocessing, and Flux Model Forward Pass

## **Project Overview**
This project demonstrates the generation of synthetic images using **Stable Diffusion**, preprocessing them for AI model input, and performing a **forward pass through a neural network built using Flux.jl**.

## **Setup Instructions**
### **1. Environment Setup**
Ensure you have access to Google Colab (recommended for GPU acceleration) and install the required dependencies by running:
```bash
!pip install diffusers transformers accelerate torch torchvision pillow numpy
```
For **Julia**, install it manually:
```bash
!wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.3-linux-x86_64.tar.gz
!tar -xvzf julia-1.9.3-linux-x86_64.tar.gz
!ln -s /content/julia-1.9.3/bin/julia /usr/local/bin/julia
```
Verify the installation:
```bash
!julia --version
```
Install required Julia packages:
```bash
!julia -e 'using Pkg; Pkg.add(["Flux", "Images", "FileIO", "NPZ"])'
```

## **2. Synthetic Image Generation**
Using **Stable Diffusion**, we generate three synthetic images based on a text prompt.
- Model used: `runwayml/stable-diffusion-v1-5`
- Images are saved as PNG files.

Run:
```python
# Define the text prompt
prompt = "a serene sunset over a futuristic city"

# Generate 3 images
for i in range(3):
    image = pipe(prompt).images[0]
    image.save(f"generated_image_{i+1}.png")
```

## **3. Image Preprocessing**
Each image is:
- Resized to **224×224** pixels
- Converted to a tensor
- Normalized to [0,1]
- Saved as a `.pt` and `.npy` file for later use.

Run:
```python
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

## **4. Flux Model Forward Pass**
A **simple neural network** is built using Flux.jl:
- **Conv2D Layer** with ReLU activation
- **MaxPooling Layer**
- **Flatten Layer**
- **Dense Layer (10 output classes)**
- **Softmax Activation**

Run the Julia script:
```julia
using Flux, NPZ

model = Chain(
    Conv((3, 3), 3 => 16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(16 * 111 * 111, 10),
    softmax
)

preprocessed_image = NPZ.npzread("preprocessed_image_1.npy")
input_image = reshape(preprocessed_image, (224, 224, 3, 1))
output = model(input_image)
println("Model output: ", output)
```
Run the script:
```bash
!julia flux_forward_pass.jl
```

## **Challenges Encountered**
1. **Stable Diffusion requires GPU**: Without a GPU, image generation is slow.
2. **Julia is not natively available in Colab**: Required manual installation.
3. **Flux model input shape mismatch**: Reshaping was necessary for compatibility.

## **Assumptions Made**
- The model expects an **RGB image in (3, 224, 224) format**.
- The project is run in **Google Colab** with GPU support.

## **Conclusion**
This project showcases the full pipeline: **image generation → preprocessing → model inference**. The combination of **Python for AI tasks** and **Julia for ML inference** provides a unique, efficient approach.

---
**Author:** Sauham Vyas
