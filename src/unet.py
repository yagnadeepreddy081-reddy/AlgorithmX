"""
SAMUDRA NETRA -- compact U-Net for oil-slick semantic segmentation.

Architecture follows Ronneberger et al. (2015) with three design changes made
for a CPU training budget:
  * base width 20 instead of 64 (parameter count ~0.6 M instead of 31 M)
  * BatchNorm after every convolution, which lets us train in far fewer epochs
  * four resolution levels instead of five, since slicks are large structures
    and the deepest level of a classic U-Net buys nothing at 256 px tiles

Output is a 5-channel logit map: sea, oil spill, look-alike, ship, land.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=5, base=20):
        super().__init__()
        b = base
        self.enc1 = DoubleConv(in_ch, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.bott = DoubleConv(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)
        self.head = nn.Conv2d(b, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bt = self.bott(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bt), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def dice_loss(logits, target, n_classes=5, eps=1.0):
    """
    Soft multi-class Dice.  Oil occupies ~1 % of pixels, so a pure
    cross-entropy objective is minimised by predicting 'sea' everywhere.
    Dice is computed per class and averaged, which makes the rare classes
    count as much as the background.
    """
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def confusion(pred, target, n_classes=5):
    """Accumulate a confusion matrix on the CPU as int64."""
    k = (target * n_classes + pred).reshape(-1)
    return torch.bincount(k, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def metrics_from_confusion(cm):
    """Per-class IoU, precision, recall from a confusion matrix (rows=truth)."""
    cm = cm.double()
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / (tp + fp + fn).clamp(min=1)
    prec = tp / (tp + fp).clamp(min=1)
    rec = tp / (tp + fn).clamp(min=1)
    return iou.tolist(), prec.tolist(), rec.tolist()
