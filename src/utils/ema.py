import torch


@torch.no_grad()
def _ema_update(teacher_mod, student_mod, momentum=0.999):
    for t_param, s_param in zip(teacher_mod.parameters(), student_mod.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)