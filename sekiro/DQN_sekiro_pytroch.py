import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, image_channels, input_dim, output_dim):
        super(DQN, self).__init__()
        # 卷积层用于处理图像输入
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=8, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层用于处理其他特征输入
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        # 最终的全连接层输出动作的Q值
        self.fc4 = nn.Linear(5632, output_dim)  # 修改为新的输入维度

    def forward(self, image_input, other_input):
        # 处理图像输入
        x_image = F.relu(self.conv1(image_input))
        x_image = self.pool1(x_image)
        x_image = F.relu(self.conv2(x_image))
        x_image = self.pool2(x_image)
        x_image = F.relu(self.conv3(x_image))
        x_image = self.pool3(x_image)
        x_image = F.relu(self.conv4(x_image))
        x_image = self.pool4(x_image)
        x_image = x_image.view(x_image.size(0), -1)  # 将卷积输出扁平化为 (batch_size, num_features)
        flatten_image = x_image.flatten()
        # 处理其他特征输入
        x_other = F.relu(self.fc1(other_input))
        x_other = F.relu(self.fc2(x_other))
        x_other = F.relu(self.fc3(x_other))
        flatten_x_other = x_other.flatten()
        # 将两个特征拼接在一起
        x_combined = torch.cat((flatten_image, flatten_x_other), dim=0)

        # 最终的全连接层输出Q值
        q_values = self.fc4(x_combined)
        return q_values



