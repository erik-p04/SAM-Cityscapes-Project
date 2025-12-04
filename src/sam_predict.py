from segment_anything import sam_model_registry, SamPredictor
import numpy as np

# Zero-shot SAM evaluation core

class SAMZeroShot:
    def __init__(self, checkpoint, model_type):
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to("cuda")
        self.predictor = SamPredictor(sam)

    def predict_with_point(self, image, point_coords, point_labels):
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        best = scores.argmax()
        return masks[best]
