import torch
import torch.nn as nn
import torchvision.transforms as tv_tfs
from PIL import Image
import os
import sys

# Import the model class from your local models.py
from models import HandmadeGlaucomaClassifier, load_compressed_checkpoint


def run_inference(image_path, model_path, device='cpu'):
    # 1. Load the pruned model using the custom loader
    # This function handles weight slicing for standalone runtime
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    model = load_compressed_checkpoint(model_path, device=device)
    model.eval()

    # 2. Setup preprocessing
    # Matches the normalization and sizing used in ImageResultsDataset
    transform = tv_tfs.Compose([
        tv_tfs.Resize((128, 128), antialias=True),
        tv_tfs.ToTensor(),
        tv_tfs.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 3. Load and process image
    try:
        img = Image.open(image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 4. Perform Inference
    with torch.no_grad():
        # Dropout is automatically disabled in eval() mode
        outputs = model(input_tensor)

    # 5. Format Results
    # Ordering: ['glaucoma', 'atrophy', 'valid_image']
    probs = outputs.cpu().numpy()[0]
    labels = ['glaucoma', 'atrophy', 'valid_image']

    results = {label: float(prob) for label, prob in zip(labels, probs)}

    print(f"\nResults for {os.path.basename(image_path)}:")
    for label, score in results.items():
        print(f" - {label:12}: {score:.4f}")

    return results


if __name__ == "__main__":
    # Example usage
    MODEL_FILE = 'artifacts/retina_nerve_classifier/x64/checkpoints/final_pruned_model.pth'
    TEST_IMAGE = 'test_retina.jpg'

    # Use GPU if available
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_inference(TEST_IMAGE, MODEL_FILE, device=DEVICE)