import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(80*70, 200)
        self.fc2 = nn.Linear(200, 3)
        self.dropout = nn.Dropout(p=0.6)
    def forward(self, x):
        x = x.view(-1, 80*70)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    dnn = DNN()
    print(dnn)
