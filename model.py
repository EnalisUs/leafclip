

import torch.nn as nn
from transformers import CLIPTextModel, RobertaModel, CLIPVisionModel
from timm import create_model
EMBEDDING_DIM = 512
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        # Load the Swin Transformer with features_only=True
        self.swin = create_model("swin_base_patch4_window7_224", pretrained=True, features_only=True)
        for param in self.swin.parameters():
            param.requires_grad = True

        # Get the feature size of the final stage
        self.swin_output_dim = self.swin.feature_info.channels()[-1]  # Last stage: 1024 channels

        # Define FC layer
        self.fc1 = nn.Linear(self.swin_output_dim * 7 * 7, EMBEDDING_DIM)  # Flattened input size
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        for param in self.fc1.parameters():
            param.requires_grad = True


    def forward(self, x):
        # Extract features from Swin
        swin_features = self.swin(x)[-1]  # Use the last stage feature map (e.g., [B, 1024, 7, 7])

        # Flatten feature map
        swin_features = swin_features.view(swin_features.size(0), -1)  # Shape: (B, 1024*7*7)

        # Pass through FC layer
        output = self.fc1(swin_features)  # Shape: (B, embedding_dim)
        return output

