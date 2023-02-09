import torch.nn as nn

class HotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = nn.Sequential(
                                            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2, stride=2, padding=0), 
                                            nn.BatchNorm2d(num_features=32),
                                            nn.ReLU())
        self.convblock2 = nn.Sequential(
                                            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), 
                                            nn.BatchNorm2d(num_features=64),
                                            nn.ReLU())
        self.convblock3 = nn.Sequential(
                                            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), 
                                            nn.BatchNorm2d(num_features=128),
                                            nn.ReLU())
        self.fc1 = nn.Sequential(
                                        nn.Linear(in_features=2048, out_features=64),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU())
        self.fc2 = nn.Sequential(
                                        nn.Linear(in_features=64, out_features=10)) #Linear는 fully connected -> 범위가 없음.. 그래서 다 더했을 때 1이 되나? 아니.? 그래서 softmax 를 취해줌
    def forward(self, x):
        B,C,H,W = x.shape
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = x.view(B, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
        