import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18
from nomeroff_net.tools.image_processing import normalize_img
from nomeroff_net.tools.mcm import modelhub, get_device_torch

device_torch = get_device_torch()


class Resnet18(object):
    def __init__(self):
        self.height = 50
        self.width = 200

        self.resnet = None
        self.model_name = "Resnet18"

    def load_model(self, path_to_model):
        resnet = resnet18(pretrained=False)
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)

        self.resnet.load_state_dict(torch.load(path_to_model, map_location=device_torch))
        self.resnet = self.resnet.to(device_torch)
        return self.resnet

    def load(self, path_to_model: str = "latest"):
        """
        TODO: describe method
        """
        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name(self.model_name)
            path_to_model = model_info["path"]
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(path_to_model,
                                                        self.model_name,
                                                        self.model_name)
            path_to_model = model_info["path"]

        return self.load_model(path_to_model)

    def preprocess(self, imgs):
        xs = []
        for img in imgs:
            x = normalize_img(img,
                              width=self.width,
                              height=self.height)
            xs.append(x)
        xs = np.moveaxis(np.array(xs), 3, 1)
        xs = torch.tensor(xs)
        xs = xs.to(device_torch)
        return xs

    @torch.no_grad()
    def forward(self, x):
        x = self.resnet(x)
        return x
