import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np

# Example usage
# image_path = 'your_image.jpg'
# image = Image.open(image_path).convert('L')
# edges_tensor = canny_edge_detection(image)
# print(edges_tensor.shape)

#
# # image = Image.open(image_path).convert('L')
# def canny_edge_detection(pil_image_greyscalse):
#     # Load image
#     image_np = np.array(pil_image_greyscalse)
#
#     # Apply Canny edge detection
#     edges = cv2.Canny(image_np, 100, 200)
#
#     # Convert edges to a tensor
#     edges_tensor = torch.tensor(edges, dtype=torch.float32)
#     edges_tensor = edges_tensor.unsqueeze(0)  # Add a channel dimension
#
#     return edges_tensor
#
#
# def sobel_edge_detection(pil_image_greyscalse):
#     # Load image
#     image_np = np.array(pil_image_greyscalse)
#
#     # Apply Sobel edge detection
#     sobel_x = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)
#     edges = np.hypot(sobel_x, sobel_y)
#     edges = (edges / edges.max() * 255).astype(np.uint8)
#
#     # Convert edges to a tensor
#     edges_tensor = torch.tensor(edges, dtype=torch.float32)
#     edges_tensor = edges_tensor.unsqueeze(0)  # Add a channel dimension
#
#     return edges_tensor
#
#
# def prewitt_edge_detection(pil_image_greyscalse):
#     # Load image
#     image_np = np.array(pil_image_greyscalse)
#
#     # Define Prewitt kernels
#     kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
#     kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
#
#     # Apply Prewitt edge detection
#     prewitt_x = cv2.filter2D(image_np, -1, kernel_x)
#     prewitt_y = cv2.filter2D(image_np, -1, kernel_y)
#     edges = np.hypot(prewitt_x, prewitt_y)
#     edges = (edges / edges.max() * 255).astype(np.uint8)
#
#     # Convert edges to a tensor
#     edges_tensor = torch.tensor(edges, dtype=torch.float32)
#     edges_tensor = edges_tensor.unsqueeze(0)  # Add a channel dimension
#
#     return edges_tensor

import torch
import kornia
from PIL import Image

def canny_edge_detection_batch(images_tensor):
    grayscale_images = None
    if images_tensor.shape[1] == 3:  # If RGB
        # Use Kornia to convert RGB to Grayscale
        grayscale_images = kornia.color.rgb_to_grayscale(images_tensor)
    elif images_tensor.shape[1] == 1:  # If already Grayscale
        grayscale_images = images_tensor
    else:
        raise ValueError("Input images must have 1 or 3 channels.")

    edges = kornia.filters.Canny()(grayscale_images)
    # Kornia's Canny returns two outputs: edges and orientation
    return edges[0]

def sobel_edge_detection_batch(images_tensor):
    grayscale_images = None
    if images_tensor.shape[1] == 3:  # If RGB
        # Use Kornia to convert RGB to Grayscale
        grayscale_images = kornia.color.rgb_to_grayscale(images_tensor)
    elif images_tensor.shape[1] == 1:  # If already Grayscale
        grayscale_images = images_tensor
    else:
        raise ValueError("Input images must have 1 or 3 channels.")

    edges = kornia.filters.sobel(grayscale_images)
    # Compute gradient magnitude from Sobel outputs
    # edges = torch.sqrt(edges[:, 0] ** 2 + edges[:, 1] ** 2)
    # edges = edges.unsqueeze(1)  # Add a channel dimension
    return edges

def prewitt_edge_detection_batch(images_tensor):
    grayscale_images = None
    if images_tensor.shape[1] == 3:  # If RGB
        # Use Kornia to convert RGB to Grayscale
        grayscale_images = kornia.color.rgb_to_grayscale(images_tensor)
    elif images_tensor.shape[1] == 1:  # If already Grayscale
        grayscale_images = images_tensor
    else:
        raise ValueError("Input images must have 1 or 3 channels.")

    kernel_x = torch.tensor([[1., 0., -1.], [1., 0., -1.], [1., 0., -1.]]).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[1., 1., 1.], [0., 0., 0.], [-1., -1., -1.]]).view(1, 1, 3, 3)

    edges_x = kornia.filters.filter2d(grayscale_images, kernel=kernel_x)
    edges_y = kornia.filters.filter2d(grayscale_images, kernel=kernel_y)
    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
    return edges

# # Example usage
# transform = transforms.Compose([
#     transforms.Grayscale(),
#     transforms.ToTensor()
# ])
#
# # Load a batch of images
# # For this example, assume 'image_paths' is a list of file paths to your images
# image_paths = ['image1.jpg', 'image2.jpg']
# images = [transform(Image.open(img_path)) for img_path in image_paths]
# images_tensor = torch.stack(images).unsqueeze(1)  # Add a channel dimension
#
# # Apply edge detection algorithms
# canny_edges = canny_edge_detection_batch(images_tensor)
# sobel_edges = sobel_edge_detection_batch(images_tensor)
# prewitt_edges = prewitt_edge_detection_batch(images_tensor)
#
# print(canny_edges.shape)
# print(sobel_edges.shape)
#print(prewitt_edges.shape)