import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        
        self.features = nn.Sequential(
            # conv3-64_1 Input:[3, 224, 224] Output:[64, 224, 224]
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # conv3-64_2 Input:[64, 224, 224] Output:[64, 224, 224]
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # pool1 Input:[64, 224, 224] Output:[64, 112, 112]
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv3-128_1 Input:[64, 112, 112] Output:[128, 112, 112]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # conv3-128_2 Input:[128, 112, 112] Output:[128, 112, 112]
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # pool2 Input:[128, 112, 112] Output:[128, 56, 56]
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv3-256_1 Input:[128, 56, 56] Output:[256, 56, 56]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # conv3-256_2 Input:[256, 56, 56] Output:[256, 56, 56]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # conv3-256_3 Input:[256, 56, 56] Output:[256, 56, 56]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # pool3 Input:[256, 56, 56] Output:[256, 28, 28]
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv3-512_1 Input:[256, 28, 28] Output:[512, 28, 28]
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # conv3-512_2 Input:[512, 28, 28] Output:[512, 28, 28]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # conv3-512_3 Input:[512, 28, 28] Output:[512, 28, 28]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # pool4 Input:[512, 28, 28] Output:[512, 14, 14]
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv3-512_4 Input:[512, 14, 14] Output:[512, 14, 14]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # conv3-512_5 Input:[512, 14, 14] Output:[512, 14, 14]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # conv3-512_6 Input:[512, 14, 14] Output:[512, 14, 14]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # pool5 Input:[512, 14, 14] Output:[512, 7, 7]
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            # fc1
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            # fc2
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            # fc3
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class TinyVGG(nn.Module):
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape, 
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1), 
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*7*7, 
                  out_features=output_shape)
    )

  def forward(self, x):
    x = self.conv_block_1(x)

    x = self.conv_block_2(x) 

    x = self.classifier(x)

    return x