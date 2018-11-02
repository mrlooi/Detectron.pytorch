from torch.nn.modules.module import Module
# from ..functions.hough_voting import HoughVotingFunction
from functions.hough_voting import HoughVotingFunction

class HoughVoting(Module):
    def __init__(self, num_classes, threshold_vote, threshold_percentage, label_threshold=500, inlier_threshold=0.9, skip_pixels=1, is_train=False):
        super(HoughVoting, self).__init__()

        self.num_classes = num_classes
        self.label_threshold = int(label_threshold)
        self.inlier_threshold = float(inlier_threshold)
        self.threshold_vote = float(threshold_vote)
        self.threshold_percentage = float(threshold_percentage)
        self.skip_pixels = int(skip_pixels)

        self.is_train = is_train

    def forward(self, label_2d, vertex_pred, extents, poses, meta_data):
        return HoughVotingFunction(self.num_classes, self.threshold_vote, self.threshold_percentage, 
        	self.label_threshold, self.inlier_threshold, self.skip_pixels, self.is_train)(label_2d, vertex_pred, extents, poses, meta_data)
