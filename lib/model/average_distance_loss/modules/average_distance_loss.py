from torch.nn.modules.module import Module
from ..functions.average_distance_loss import AverageDistanceLossFunction

class AverageDistanceLoss(Module):
    def __init__(self, num_classes, margin):
        super(AverageDistanceLoss, self).__init__()

        self.num_classes = int(num_classes)
        self.margin = float(margin)

    def forward(self, poses_pred, poses_target, poses_weight, points, symmetry):
        return AverageDistanceLossFunction(self.num_classes, self.margin)(poses_pred, poses_target, poses_weight, points, symmetry)
