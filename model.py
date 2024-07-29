import torch.nn as nn
import torch
from Global_Dropout import Global_Dropout

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.glob_drop = Global_Dropout(p1 = 0.3, p2 = 0.3)
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Flatten(),
            nn.Linear(40000, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5)
            )
        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Flatten(),
            nn.Linear(40000, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5)
            )
        self.features3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Flatten(),
            nn.Linear(9248, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5)
            )
        self.fc = nn.Sequential(
            nn.Linear(128*3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.25)
            )
        self.classification = nn.Sequential(
            #nn.Dropout(p=0.5),
            nn.Linear(128, 43),
            nn.Sigmoid()
            )
        
    def forward(self, x1, x2, x3):
        x1, x2, x3 = self.glob_drop(x1, x2, x3)
        x1 = self.features1(x1)
        x2 = self.features2(x2)
        x3 = self.features3(x3)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fc(x)
        x = self.classification(x)
        return x
    

    
    
    
    
    
    
    
    
    
    
    