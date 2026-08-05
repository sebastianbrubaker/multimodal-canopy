import torch
import torch.nn as nn



# =============================================================================
# U-Net Utilities
# =============================================================================



class DoubleConv(nn.Module):
    def __init__(self, n_in_channels, n_out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_in_channels, n_out_channels, 3, 1, 1, bias=False),    # no bias as we use batch normalization
            nn.BatchNorm2d(n_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_out_channels, n_out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(n_out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.conv(x)


class Contract(nn.Module):
    def __init__(self, channel_counts=[3, 64, 128, 256]):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.convs = nn.ModuleList()
        for i in range(len(channel_counts) - 1):
            n_in = channel_counts[i]
            n_out = channel_counts[i+1]

            self.convs.append(DoubleConv(n_in, n_out))


    def forward(self, x):
        skips = []
        for conv in self.convs:
            x = conv(x)
            skips.append(x)    # accumulate hidden features tensors for skip connections
            x = self.pool(x)    # pool every layer except the bottom

        return x, skips


class Expand(nn.Module):
    def __init__(self, n_encoders, channel_counts=[256, 128, 64]):
        super().__init__()

        self.upconvs = nn.ModuleList()
        self.convs = nn.ModuleList()

        for i in range(len(channel_counts) - 1):
            n_in = channel_counts[i]
            n_out = channel_counts[i+1]

            self.upconvs.append(nn.ConvTranspose2d(n_in, n_in, 2, 2))

            total_n_in = n_in + (n_in * n_encoders)    # get n channels after concatenation
            self.convs.append(DoubleConv(total_n_in, n_out))


    def forward(self, x, all_encoder_skips):
        for i, (upconv, conv) in enumerate(zip(self.upconvs, self.convs)):
            # Select skips for current level 
            level_skips = [encoder_skips[-(1+i)] for encoder_skips in all_encoder_skips]

            # Upconv, concat, convolve
            x = upconv(x)
            x = torch.cat([x] + level_skips, dim=1)
            x = conv(x)

        return x



# =============================================================================
# U-Net Model(s)
# =============================================================================



class DualEncoderUNet(nn.Module):
    def __init__(self, n_sar_channels=2, n_opt_channels=9, n_out_channels=1):
        super().__init__()

        # Define encoding path
        hidden_channels = [64, 128, 256]
        sar_channel_counts = [n_sar_channels] + hidden_channels
        opt_channel_counts = [n_opt_channels] + hidden_channels

        self.sar_enc = Contract(sar_channel_counts)
        self.opt_enc = Contract(opt_channel_counts)

        # Define bottleneck
        bottleneck_in = hidden_channels[-1] * 2
        bottleneck_out = hidden_channels[-1]
        self.bottleneck_conv = DoubleConv(bottleneck_in, bottleneck_out)

        # Define decoding path
        self.dec = Expand(2, hidden_channels[::-1])

        # Define final convolution
        self.final_conv = nn.Conv2d(hidden_channels[0], n_out_channels, 1, 1)


    def forward(self, sar, opt):
        # Encoding path
        s, s_skips = self.sar_enc(sar)
        o, o_skips = self.opt_enc(opt)

        # Bottleneck and fusion
        bottleneck = self.bottleneck_conv(torch.cat([s, o], dim=1))

        # Decoding path
        x = self.dec(bottleneck, [s_skips, o_skips])

        return self.final_conv(x)