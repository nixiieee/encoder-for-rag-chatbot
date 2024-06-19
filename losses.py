import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-6)

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def calc_cosine(self,x1,x2):
        return self.cosine(x1,x2)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative_a = self.calc_euclidean(anchor, negative)
        distance_negative_b = self.calc_euclidean(positive, negative)
        
        losses = torch.relu(distance_positive - (distance_negative_a + distance_negative_b)/2.0 + self.margin)

        return losses.mean()

class SoftmaxLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(SoftmaxLoss, self).__init__()
        self.alpha=alpha

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        pos_sim = torch.sum(anchor * positive, dim=1, keepdim=True)
        neg_mul = torch.matmul(anchor, negative.t())
        neg_sim = torch.logsumexp(neg_mul, dim=1, keepdim=True)

        loss = torch.relu(neg_sim - pos_sim + self.alpha)

        return loss.mean()
    
class SoftmaxLossRScore(nn.Module):
    def __init__(self, alpha=1.0):
        super(SoftmaxLossRScore, self).__init__()
        self.alpha=alpha

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, r_score) -> torch.Tensor:
        pos_sim = torch.sum(anchor * positive, dim=1, keepdim=True)
        neg_mul = torch.matmul(anchor, negative.t())
        neg_sim = torch.logsumexp(neg_mul, dim=1, keepdim=True)

        loss = torch.relu(neg_sim - pos_sim + r_score + self.alpha)

        return loss.mean()
    
class TripletMarginLossRScore(nn.Module):
    def __init__(self, margin = 1.0):
        super(TripletMarginLossRScore, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return(x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, r_score) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        
        losses = torch.relu(distance_positive - distance_negative + self.margin + r_score)

        return losses.mean()