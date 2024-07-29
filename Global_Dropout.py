import torch.nn as nn
import torch
import random

class Global_Dropout(nn.Module):
    def __init__(self, p1, p2):
        super().__init__()
        self.p1, self.p2 = p1, p2
        
    def forward(self, x1, x2, x3):
        if self.training:
            for i in range(len(x1)):
                p = random.choices(range(3), weights=[self.p1, self.p2, 
                                                      1 - self.p1 - self.p2])
                if p[0] == 0:
                    x1[i] = torch.zeros_like(x1[i])
                if p[0] == 1:
                    x2[i] = torch.zeros_like(x2[i])
            
        return x1, x2, x3
    
