'''
  Adapted from here: https://github.com/ZitongYu/PhysFormer/blob/main/TorchLossComputer.py
  Modifed based on the HR-CNN here: https://github.com/radimspetlik/hr-cnn
'''
import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb
import torch.nn as nn

def normal_sampling(mean, label_k, std):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

def kl_loss(inputs, labels):
    # Reshape the labels tensor to match the shape of inputs
    labels = labels.view(1, -1)
    
    # Compute the KL Div Loss
    criterion = nn.KLDivLoss(reduction='sum')
    loss = criterion(F.log_softmax(inputs, dim=-1), labels)
    return loss
 
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Implementation of Focal Loss.
        Args:
            gamma (float): Focusing parameter. Higher values give more weight to
                           hard-to-classify examples.
            alpha (float, optional): Weighting factor for the positive class.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (Tensor) Model logits, shape (N, C)
            targets: (Tensor) Ground truth labels, shape (N)
        """
        # Calculate Cross Entropy Loss, but without reduction
        # This gives us -log(pt)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get pt (probability of the correct class)
        # pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # Calculate the Focal Loss
        # FL = (1 - pt)^gamma * CE_Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            # Assuming binary or multi-class with alpha per class
            # This part might need adjustment based on your specific 'alpha' needs
            # For simple alpha (e.g., binary), you might do:
            # alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
            # focal_loss = alpha_t * focal_loss
            # ---
            # For now, we'll stick to the common form without alpha if it's complex
            # Or assume alpha is a tensor of weights per class
            if isinstance(self.alpha, (float, int)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device)
                focal_loss = torch.where(targets == 1, alpha_t * focal_loss, (1 - alpha_t) * focal_loss)
            elif torch.is_tensor(self.alpha):
                alpha_t = self.alpha.gather(0, targets.data.view(-1)).to(inputs.device)
                focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TorchLossComputer(object):
    @staticmethod
    def compute_complex_absolute_given_k(output, k, N):
        two_pi_n_over_N = torch.autograd.Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
        hanning = torch.autograd.Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        k = k.type(torch.FloatTensor).cuda()
        two_pi_n_over_N = two_pi_n_over_N.cuda()
        hanning = hanning.cuda()
            
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                            + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

        return complex_absolute

    @staticmethod
    def complex_absolute(output, Fs, bpm_range=None):
        output = output.view(1, -1)

        N = output.size()[1]

        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz

        # only calculate feasible PSD range [0.7,4] Hz
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N)

        return (1.0 / complex_absolute.sum()) * complex_absolute	# Analogous Softmax operator      
        
    @staticmethod
    def cross_entropy_power_spectrum_loss(inputs, target, Fs):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

    @staticmethod
    def cross_entropy_power_spectrum_focal_loss(inputs, target, Fs, gamma):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        #pdb.set_trace()
        criterion = FocalLoss(gamma=gamma)

        return criterion(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

        
    @staticmethod
    def cross_entropy_power_spectrum_forward_pred(inputs, Fs):
        inputs = inputs.view(1, -1)
        bpm_range = torch.arange(40, 190, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        return whole_max_idx
    
    @staticmethod
    def cross_entropy_power_spectrum_DLDL_softmax2(inputs, target, Fs, std):
        target_distribution = [normal_sampling(int(target), i, std) for i in range(40, 180)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(torch.device('cuda'))
        
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        
        bpm_range = torch.arange(40, 180, dtype=torch.float).to(torch.device('cuda'))

        ca = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)
        
        fre_distribution = ca/torch.sum(ca)
        loss_distribution_kl = kl_loss(fre_distribution, target_distribution)
        
        whole_max_val, whole_max_idx = ca.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        return loss_distribution_kl, F.cross_entropy(ca, (target-bpm_range[0]).view(1).type(torch.long)),  torch.abs(target[0]-bpm_range[0]-whole_max_idx)
        