import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.relu,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.relu,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self.relu,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.relu,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.relu,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 128),
            self.relu,
            nn.Linear(128, 128),
            self.relu,
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x