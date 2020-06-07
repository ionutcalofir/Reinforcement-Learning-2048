import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size=192):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_size, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 100)

        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 4)

        self.fc6 = nn.Linear(100, 60)
        self.fc7 = nn.Linear(60, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        a = F.relu(self.fc4(x))
        a = self.fc5(a)

        s = F.relu(self.fc6(x))
        s = self.fc7(s)

        return s + (a - torch.mean(a))
