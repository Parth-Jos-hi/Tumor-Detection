import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from model import BrainTumorCNN


# -----------------------------
# 1. LOAD MODEL
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BrainTumorCNN().to(device)
model.load_state_dict(torch.load("brain_tumor_cnn.pth", map_location=device))
model.eval()


# -----------------------------
# 2. IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    tensor = torch.tensor(img, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return img, tensor.to(device)


# -----------------------------
# 3. GRAD-CAM
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, img_tensor):
        output = self.model(img_tensor)
        self.model.zero_grad()
        output.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam / cam.max()

        return cam


gradcam = GradCAM(model, model.conv3)
# 4. STREAMLIT UI
st.title("🧠 Brain Tumor CNN – Model Inspection Tool")
st.write("Upload an MRI image to inspect the CNN prediction and attention.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

THRESHOLD = 0.7  # 🔑 key fix to reduce false positives

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    img, img_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(img_tensor).item()

    prediction = "Tumor" if output > THRESHOLD else "No Tumor"
    confidence = output if output > THRESHOLD else 1 - output

    cam = gradcam.generate(img_tensor)
    # -------------------------
    # DISPLAY RESULTS
    # -------------------------
    st.subheader("Prediction")
    st.write(f"**Class:** {prediction}")
    st.write(f"**Confidence:** {confidence:.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].set_title("Original MRI")
    ax[0].imshow(img, cmap="gray")
    ax[0].axis("off")

    ax[1].set_title("Grad-CAM")
    ax[1].imshow(img, cmap="gray")
    ax[1].imshow(cam, cmap="jet", alpha=0.5)
    ax[1].axis("off")

    st.pyplot(fig)
