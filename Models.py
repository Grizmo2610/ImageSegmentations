import torch
import torch.nn as nn
from torchvision import transforms


class DoubleConv(nn.Module):
    """
    A block of two convolutional layers with ReLU activation and padding.

    Attributes:
        conv (nn.Sequential): Sequential container for two convolutional layers.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initializes the DoubleConv block with two convolutional layers.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
        """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DoubleConv block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for image segmentation.

    Attributes:
        enc1, enc2, enc3, enc4 (DoubleConv): Encoder blocks for downsampling.
        pool (nn.MaxPool2d): Max pooling layer.
        bottleneck (DoubleConv): Bottleneck layer at the bottom of the network.
        up1, up2, up3, up4 (nn.ConvTranspose2d): Transpose convolutions for upsampling.
        dec1, dec2, dec3, dec4 (DoubleConv): Decoder blocks for upsampling.
        final (nn.Conv2d): Final convolutional layer to output the segmentation map.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 91) -> None:
        """
        Initializes the U-Net model.

        Args:
            in_channels (int): Number of input channels (default is 3 for RGB images).
            num_classes (int): Number of output classes (default is 91 for COCO dataset).
        """
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width).
        """
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        up4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))
        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        return self.final(dec1)


TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize input image to 224x224 pixels
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for RGB input
])


def load_model(device: str = 'cpu', model_path: str = 'models/best_unet.pth') -> nn.Module:
    """
    Loads a pre-trained U-Net model.

    Args:
        device (str): The device to load the model on ('cpu' or 'cuda').
        model_path (str): The path to the pre-trained model weights.

    Returns:
        nn.Module: The loaded U-Net model.
    """
    device = torch.device(device)
    model = UNet().to(device)
    map_location = device
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=map_location))
    model.eval()  # Set the model to evaluation mode
    print(f'App running on device: {device}')
    return model