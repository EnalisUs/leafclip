
from transformers import AutoTokenizer
import json
import torch

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
device = "cpu"
def load_captions_and_features(path="leafclip_captions_features.pth"):
    data = torch.load(path)
    captions = data["captions"]
    text_features = data["text_features"].to(device)
    return captions, text_features

def predict_image_v2(model, image_path, meta_path, top_k=5):
    """
    Predicts the best-matching caption(s) for a single image using the trained vision-language model.

    Args:
        model: The trained vision-language model.
        image_path: Path to the input image.
        meta_path: Path to the JSON metadata file containing captions.
        top_k: Number of top predictions to return.
    Returns:
        top_k_predictions: List of top-k captions with their similarity scores.
    """
    model.eval()

    # Load and process the image
    image = Image.open(image_path).convert("RGB")
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = image_transforms(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    # Get captions and their features
    captions, all_text_features = load_captions_and_features(path="leafclip_captions_features.pth")

    with torch.no_grad():
        image_features = model(image)
        image_features = F.normalize(image_features, p=2, dim=-1)

    # Normalize features
    image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
    text_features_norm = all_text_features / all_text_features.norm(dim=1, keepdim=True)

    # Compute similarity scores and convert to probabilities
    similarity_scores = torch.matmul(image_features_norm, text_features_norm.T).squeeze(0)
    similarity_scores = similarity_scores / 0.01
    probabilities = F.softmax(similarity_scores, dim=0)

    # Get top-k predictions
    top_k_indices = torch.topk(probabilities, k=top_k).indices
    top_k_indices = top_k_indices.cpu().numpy()
    
    top_k_predictions = [(captions[idx], probabilities[idx].item()) for idx in top_k_indices]
    return top_k_predictions
