import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from model import BrainTumorCNN
# LOAD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainTumorCNN().to(device)
model.load_state_dict(torch.load("brain_tumor_cnn.pth", map_location=device))
model.eval()

# LOAD & PREPROCESS IMAGE
def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img_tensor = torch.tensor(img, dtype=torch.float32)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 224, 224]
    return img, img_tensor.to(device)

# OCCLUSION SENSITIVITY
def occlusion_sensitivity(img, img_tensor, patch_size=32):
    heatmap = np.zeros((224, 224))
    with torch.no_grad():
        original_pred = model(img_tensor).item()
    for y in range(0, 224, patch_size):
        for x in range(0, 224, patch_size):
            occluded = img.copy()
            occluded[y:y + patch_size, x:x + patch_size] = 0
            occluded_tensor = torch.tensor(
                occluded, dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(occluded_tensor).item()
            heatmap[y:y + patch_size, x:x + patch_size] = original_pred - pred
    return heatmap

# GRAD-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
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

# RUN ANALYSIS
image_path = "C:/Users/pc/ML/Models/Tumor/Data/yes/Y59.jpg"
img, img_tensor = load_image(image_path)
occ_map = occlusion_sensitivity(img, img_tensor)
gradcam = GradCAM(model, model.conv3)
cam = gradcam.generate(img_tensor)

# VISUALIZATION
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Occlusion Sensitivity")
plt.imshow(occ_map, cmap="hot")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Grad-CAM")
plt.imshow(img, cmap="gray")
plt.imshow(cam, cmap="jet", alpha=0.5)
plt.axis("off")

plt.show()
