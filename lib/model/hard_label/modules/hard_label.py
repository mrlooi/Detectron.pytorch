from torch.nn.modules.module import Module
from ..functions.hard_label import HardLabelFunction

class HardLabel(Module):
    def __init__(self, threshold):
        super(HardLabel, self).__init__()

        self.threshold = float(threshold)

    def forward(self, label_gt, prob):
        return HardLabelFunction(self.threshold)(label_gt, prob)
