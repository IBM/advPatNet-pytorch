import torch
from torch import nn

'''
class NLL_OHEM(th.nn.NLLLoss):
    """ Online hard example mining. 
    Needs input from nn.LogSotmax() """

    def __init__(self, ratio):
        super(NLL_OHEM, self).__init__(None, True)
        self.ratio = ratio

    def forward(self, x, y, ratio=None):
        if ratio is not None:
            self.ratio = ratio
        num_inst = x.size(0)
        num_hns = int(self.ratio * num_inst)
        x_ = x.clone()
        inst_losses = th.autograd.Variable(th.zeros(num_inst)).cuda()
        for idx, label in enumerate(y.data):
            inst_losses[idx] = -x_.data[idx, label]
            # loss_incs = -x_.sum(1)                                                    
        _, idxs = inst_losses.topk(num_hns)
        x_hn = x.index_select(0, idxs)
        y_hn = y.index_select(0, idxs)
        return th.nn.functional.nll_loss(x_hn, y_hn)     
'''

class Adv_OHEM(nn.Module):
    """ Online hard example mining.
    Needs the max probability of the bboxes in each image  """

    def __init__(self, ratio):
        super(Adv_OHEM, self).__init__()
        self.ratio = ratio

    def forward(self, x, ratio=None):
        if ratio is not None:
            self.ratio = ratio
        num_inst = x.size(0)
        num_hns = int(self.ratio * num_inst)
        #x_ = x.clone()
        inst_losses = torch.autograd.Variable(torch.zeros(num_inst)).cuda()
        for idx, prob in enumerate(x.data):
            inst_losses[idx] = x.data[idx]
            # loss_incs = -x_.sum(1)

        _, idxs = inst_losses.topk(num_hns)
        x_hn = x.index_select(0, idxs)
        return torch.mean(x_hn)
