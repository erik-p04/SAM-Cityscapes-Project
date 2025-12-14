import torch
from segment_anything import sam_model_registry

class SAMBackbone(torch.nn.Module):
    def __init__(self, checkpoint, model_type="vit_h"):
        super().__init__()

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.image_encoder = sam.image_encoder
        self.image_encoder.eval()

        for p in self.image_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        """
        x: Tensor [B, 3, H, W] (normalized RGB)
        returns: feature map [B, C, H/16, W/16]
        """
        return self.image_encoder(x)
