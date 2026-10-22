"""Preserve the notebook's two PyTorch architectures as explicit reference runs."""


def build_reference(config: dict):
    """Construct reference pooling variants at the configured, actual input shape."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class CNN(nn.Module):
        """Retain the notebook backbone so new families have a recognizable control."""

        def __init__(self):
            """Infer flatten dimensions instead of retaining stale 15-frame assumptions."""
            super().__init__()
            self.pool_after_conv3 = config['family'] == 'reference_a'
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(config['dropout'])
            with torch.no_grad():
                size = self._features(torch.zeros(1, 1, config['features']['frames'],
                                                 config['features']['coefficients'])).flatten(1).shape[1]
            self.fc1 = nn.Linear(size, 64); self.fc2 = nn.Linear(64, 1)

        def _features(self, x):
            """Isolate pooling choice while keeping the reference convolution flow intact."""
            x = self.dropout(self.pool(F.relu(self.conv1(x))))
            x = self.dropout(self.pool(F.relu(self.conv2(x))))
            x = F.relu(self.conv3(x))
            if self.pool_after_conv3:
                x = self.pool(x)
            return self.dropout(x)

        def forward(self, x):
            """Return raw logits for numerically stable binary loss computation."""
            x = self._features(x).flatten(1)
            return self.fc2(self.dropout(F.relu(self.fc1(x))))

    return CNN()
