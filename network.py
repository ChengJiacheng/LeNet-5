import torch.nn as nn

class LeNet(nn.Module):

    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.relu(x)
        x = self.sigmoid(x)

        x = self.maxpool(x)
        x = self.conv2(x)
        # x = self.relu(x)
        x = self.sigmoid(x)

        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.relu(x)
        x = self.sigmoid(x)

        x = self.fc2(x)
        # x = self.relu(x)
        x = self.sigmoid(x)

        x = self.fc3(x)

        return x
    
    
class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(32 * 32, 20)
        self.fc3 = nn.Linear(20, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.sigmoid(x)

        x = self.fc3(x)

        return x