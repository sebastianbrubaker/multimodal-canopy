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
        n_in = channel_counts[0] 
        for n_out in channel_counts[1:]:
            self.convs.append(DoubleConv(n_in, n_out))
            n_in = n_out


    def forward(self, x):
        skips = []
        for i, conv in enumerate(self.convs):
            x = conv(x)
            skips.append(x)    # accumulate hidden features tensors for skip connections

            if i < len(self.convs) - 1: x = self.pool(x)    # pool every layer except the bottom

        return x, skips


class Expand(nn.Module):
    def __init__(self, n_encoders, channel_counts=[256, 128, 64, 1]):
        super().__init__()

        self.upconvs = nn.ModuleList()
        self.convs = nn.ModuleList()

        n_in = channel_counts[0]
        for n_out in channel_counts[1:]:
            self.upconvs.append(nn.ConvTranspose2d(n_in, n_out, 2, 2))

            n_in = n_out + (n_out * n_encoders)    # update for DoubleConv instance

            self.convs.append(DoubleConv(n_in, n_out))

            n_in = n_out    # update for ConvTransposed2d instance


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
# U-Net Models
# =============================================================================



class DualEncoderUNet(nn.Module):
    def __init__(self, n_sar_channels=2, n_opt_channels=9, n_out_channels=1):
        super.__init__()

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
        dec_channel_counts = hidden_channels[::-1] + [n_out_channels]
        self.dec = Expand(2, dec_channel_counts)


    def forward(self, sar, opt):
        # Encoding path
        s, s_skips = self.sar_enc(sar)
        o, o_skips = self.opt_enc(opt)

        # Bottleneck and fusion
        bottleneck = self.bottleneck_conv(torch.cat([s, o], dim=1))

        # Decoding path
        x = self.dec(bottleneck, [s_skips, o_skips])

        return x