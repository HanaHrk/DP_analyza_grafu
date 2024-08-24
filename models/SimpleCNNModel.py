import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNNModel(torch.nn.Module):
    def __init__(self, device, out_dim, dropout):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.dropout = dropout
        print(f"Running on {self.device}")
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3, 3))
        self.conv_layer2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3))
        self.pooling = nn.MaxPool2d(kernel_size=(5, 5))

        # output shape = (batch_size, 32, 1, 50)
        self.dropout = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(78408, 1000 // 2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1000 // 2, out_dim)

        self.to(self.device)
        # Progresses data across layers

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.pooling(out)
        out = self.dropout(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)

        return out

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
