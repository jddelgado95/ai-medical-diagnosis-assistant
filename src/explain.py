import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

def grad_cam(model, image_tensor, target_layer):
    model.eval()
    gradients = []
    activations = []

    def save_gradient(module, input, output):
        gradients.append(output[0].detach())

    def save_activation(module, input, output):
        activations.append(output.detach())

    handle_activ = target_layer.register_forward_hook(save_activation)
    handle_grad = target_layer.register_full_backward_hook(save_gradient)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1)
    model.zero_grad()
    output[0, pred_class].backward()

    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    activ = activations[0][0]
    for i in range(activ.shape[0]):
        activ[i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activ, dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap


def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    return np.uint8(superimposed_img)