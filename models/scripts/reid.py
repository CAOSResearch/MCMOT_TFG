import cv2
import torch
import torch.nn as nn
import numpy as np

from PIL import Image

import torchvision.transforms as T
from torchvision import models


class ReIDExtractor:

    def __init__(
        self,
        backbone="resnet50",
        input_size=(224, 224),
        device="cuda"
    ):

        self.device = (
            device
            if torch.cuda.is_available()
            else "cpu"
        )

        self.input_size = input_size

        # ReID backbone selection
        if backbone.lower() == "resnet50":

            self.model = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2
            )

        elif backbone.lower() == "resnet101":

            self.model = models.resnet101(
                weights=models.ResNet101_Weights.IMAGENET1K_V2
            )

        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}"
            )

        self.model.fc = nn.Identity()

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        # Preprocessing transformations
        self.transform = T.Compose([
            T.Resize(self.input_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _preprocess(
        self,
        crop
    ):

        if isinstance(
            crop,
            np.ndarray
        ):

            crop = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            )

            crop = Image.fromarray(
                crop
            )

        return self.transform(
            crop
        )
    
    # Batch extraction of features
    def extract_batch(
        self,
        crops
    ):

        if len(crops) == 0:

            feat_dim = (
                2048
            )

            return np.empty(
                (0, feat_dim),
                dtype=np.float32
            )

        tensors = [
            self._preprocess(c)
            for c in crops
        ]

        x = torch.stack(
            tensors
        ).to(self.device)

        with torch.inference_mode():

            if (
                self.device == "cuda"
            ):

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    feats = self.model(x)

            else:

                feats = self.model(x)

        feats = feats.float()

        feats = feats / (
            feats.norm(
                p=2,
                dim=1,
                keepdim=True
            ) + 1e-12
        )

        return (
            feats
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def extract(
        self,
        crop
    ):

        return self.extract_batch(
            [crop]
        )[0]