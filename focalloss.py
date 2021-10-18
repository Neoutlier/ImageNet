import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input2, target):
        #print(input2.shape)
        #print(target.shape)
        input2 = input2.permute(0, 2, 1)
        if input2.dim()>2:
            input2 = input2.view(-1, input2.size(-1))
        target = target.view(-1,1)

        logpt = F.log_softmax(input2)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input2.data.type():
                self.alpha = self.alpha.type_as(input2.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
