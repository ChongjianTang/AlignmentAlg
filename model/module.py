import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================================================================


class OMNetEncoder(nn.Module):
    """
    Feature Extraction 1
    """

    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))

        self.conv_block2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))

        self.conv_block3 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))

        self.conv_block4 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True))

        self.conv_block5 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True))

    def forward(self, x, mask=None):
        x = self.conv_block1(x)
        point_feat64 = self.conv_block2(x)
        x = self.conv_block3(point_feat64)
        x = self.conv_block4(x)
        point_feat1024 = self.conv_block5(x)

        if mask is None:
            L = [point_feat64, point_feat1024]
            glob_feat = torch.max(point_feat1024, dim=-1, keepdim=True)[0]
        else:
            L = [point_feat64 * mask, point_feat1024 * mask]
            glob_feat = torch.max(point_feat1024 * mask, dim=-1, keepdim=True)[0]

        return L, glob_feat


class OMNetFusion(nn.Module):
    """
    Feature Extraction 2
    """

    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(nn.Conv1d(2048 + 64, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True))

        self.conv_block2 = nn.Sequential(nn.Conv1d(1024, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True))

        self.conv_block3 = nn.Sequential(nn.Conv1d(1024, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True))

    def forward(self, x, mask=None):

        point_feat1024_0 = self.conv_block1(x)
        point_feat1024_1 = self.conv_block2(point_feat1024_0)
        fuse_feat = self.conv_block3(point_feat1024_1)

        if mask is None:
            L = [point_feat1024_0, point_feat1024_1]
            fuse_feat = fuse_feat

        else:
            L = [point_feat1024_0 * mask, point_feat1024_1 * mask]
            fuse_feat = fuse_feat * mask

        return L, fuse_feat


class OMNetDecoder(nn.Module):
    """
    Mask Prediction
    """

    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True))

        self.conv_block2 = nn.Sequential(nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True))

        self.conv_block3 = nn.Sequential(nn.Conv1d(256, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True))

        self.conv_block4 = nn.Sequential(nn.Conv1d(256, 2, 1))

    def forward(self, x):
        point_feat512 = self.conv_block1(x)
        point_feat256_0 = self.conv_block2(point_feat512)
        point_feat256_1 = self.conv_block3(point_feat256_0)
        L = [point_feat512, point_feat256_0, point_feat256_1]
        cls = self.conv_block4(point_feat256_1)

        return L, cls


class OMNetRegression(nn.Module):
    """
    Transformation Regression
    """

    def __init__(self):
        super().__init__()
        self.fc_block1 = nn.Sequential(
            nn.Linear(3072, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True)
        )

        self.fc_block2 = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True))

        self.fc_block3 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True))

        self.fc_block4 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True))

        self.final_fc = nn.Sequential(nn.Linear(256, 7))

    def forward(self, x):
        x = self.fc_block1(x)
        x = self.fc_block2(x)
        x = self.fc_block3(x)
        x = self.fc_block4(x)
        pred_pose = self.final_fc(x)
        pred_quat, pred_translate, = pred_pose[:, :4], pred_pose[:, 4:]
        pred_quat = F.normalize(pred_quat, dim=1)
        pred_pose = torch.cat((pred_quat, pred_translate), dim=1)  # (B, 7)

        return pred_pose
