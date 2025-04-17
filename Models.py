import torch
import torch.nn as nn
from torchvision import transforms

class AttentionGate(nn.Module):
    """
    Attention Gate to refine skip connections in the decoder path.
    Enhances relevant features from the encoder using decoder context.

    Args:
        F_g (int): Number of channels in the gating signal (from decoder).
        F_l (int): Number of channels in the input feature map (from encoder).
        F_int (int): Number of intermediate channels for attention computation.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, g, x):
        """
        Forward pass of the attention gate.

        Args:
            g (Tensor): Decoder features (gating signal).
            x (Tensor): Encoder features to be refined.

        Returns:
            Tensor: Refined encoder features after applying attention.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    """
    Double convolution block with optional dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dropout (bool): Whether to apply dropout after convolutions.
    """
    def __init__(self, in_channels, out_channels, dropout=False):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(p=0.5))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the double convolution block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after convolutions.
        """
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture with Attention Gates for image segmentation.

    Args:
        in_channels (int): Number of input image channels. Default is 3 (RGB).
        num_classes (int): Number of output segmentation classes.
    """
    def __init__(self, in_channels=3, num_classes=91):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024, dropout=True)

        # Decoder + Attention
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionGate(512, 512, 256)
        self.dec4 = DoubleConv(1024, 512, dropout=True)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(256, 256, 128)
        self.dec3 = DoubleConv(512, 256, dropout=True)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(128, 128, 64)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(64, 64, 32)
        self.dec1 = DoubleConv(128, 64)

        # Output layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for Conv and ConvTranspose layers using Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Segmentation map of shape (B, num_classes, H, W).
        """
        # Encoding path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoding path with attention
        up4 = self.up4(bottleneck)
        enc4_att = self.att4(up4, enc4)
        dec4 = self.dec4(torch.cat([up4, enc4_att], dim=1))

        up3 = self.up3(dec4)
        enc3_att = self.att3(up3, enc3)
        dec3 = self.dec3(torch.cat([up3, enc3_att], dim=1))

        up2 = self.up2(dec3)
        enc2_att = self.att2(up2, enc2)
        dec2 = self.dec2(torch.cat([up2, enc2_att], dim=1))

        up1 = self.up1(dec2)
        enc1_att = self.att1(up1, enc1)
        dec1 = self.dec1(torch.cat([up1, enc1_att], dim=1))

        return self.final(dec1)


TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize RGB
])


def load_model(device: str = 'cpu', model_path: str = 'models/best_unet.pth') -> nn.Module:
    """
    Load a pre-trained U-Net model from disk.

    Args:
        device (str): Device to load the model onto ('cpu' or 'cuda').
        model_path (str): Path to the saved model weights.

    Returns:
        nn.Module: Loaded and ready-to-infer U-Net model.
    """
    device = torch.device(device)
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()
    print(f'App running on device: {device}')
    return model