import torch.nn as nn
import torch.nn.functional as F

def criterion_KD(
    outputs,
    label,
    teacher_outputs,
    alpha: float = 0.9,
    temperature: float = 3.
):
    loss_KD = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(outputs / temperature, dim=1),
        F.softmax(teacher_outputs / temperature, dim=1)
    ) * (alpha * temperature * temperature) + \
        F.cross_entropy(outputs, label) * (1. - alpha)
    return loss_KD