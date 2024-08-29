import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetModel(torch.nn.Module):
    def __init__(self, device, use_pretrained_weights, out_dim):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"Running on {self.device}")
        self.use_pretrained_weights = use_pretrained_weights
        print(f"Use pretrained weights: {self.use_pretrained_weights}")
        # Let's load pre-trained model ResNet-152
        self.model = models.resnet152(pretrained=True if self.use_pretrained_weights else False)

        # add fully-connected output layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, out_dim)

        self.to(self.device)
        # Progresses data across layers

    def forward(self, x):
        x = self.model(x)
        return x

    def save_learnable_params(self, path):
        sd = self.state_dict()
        rmkeys = [name for name, param in self.named_parameters() if not param.requires_grad]
        for k in rmkeys:
            sd.pop(k)
        torch.save(sd, path)

    def load_learnable_params(self, path):
        mk = self.load_state_dict(torch.load(path, map_location=self.device), strict=False)
        for k in mk.missing_keys:
            assert k.startswith("backbone.")

    def get_number_of_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_number_of_all_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    max_input_len = 400
    random_data = torch.rand((1, 3, 500, 500))
    model = SimpleCNNModel(device="cpu", out_dim=15, dropout=0.1)
    result = model(random_data)
    print(result)
