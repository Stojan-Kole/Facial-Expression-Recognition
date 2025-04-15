import torch.nn.functional as F
import torch.nn as nn

class EmotionRecognition(nn.Module):

    def __init__(self):
        super(EmotionRecognition, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(in_features=64 * 12 * 12, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=7)

    def forward(self, x):
        #Convolution + Relu + Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 12 * 12)

        x = F.relu(self.fc1(x))
        #this should help with overfitting
        x = self.dropout(x)
        x = self.fc2(x)


        return x
