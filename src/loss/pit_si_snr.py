import torch
from torch import nn

class PITSiSNRLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def sisnr(self, pred: torch.Tensor, ref: torch.Tensor, eps=1e-8):
        """
        Calculate scale-invariant signal-to-noise ratio (SI-SNR) in a tensorized form.
        Args:
            pred: separated signal, B x L tensor
            ref: reference signal, B x L tensor
        Returns:
            SI-SNR value as B tensor
        """
        pred_mn = pred - pred.mean(dim=-1, keepdim=True)
        ref_mn = ref - ref.mean(dim=-1, keepdim=True)
        t = (torch.sum(pred_mn * ref_mn, dim=-1, keepdim=True) * ref_mn) / (ref_mn.norm(dim=-1, keepdim=True) ** 2 + eps)
        return 20 * torch.log10(eps + t.norm(dim=-1) / (pred_mn - t).norm(dim=-1) + eps)
    


    def forward(self, output_audios: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor, **batch):
        """
        Loss function calculation logic.

        Calculate permutation-invariant SI-SNR loss for source separation.
        Args:
            output_audios: Predicted separated signals, tensor (num_spks, B, L)
            s1: Reference signal 1, tensor (num_spks, B, L)
            s2: Reference signal 2, tensor (num_spks, B, L)
        Returns:
            Permutation-invariant SI-SNR loss (scalar)
        """
        egs = torch.stack([s1, s2])
        num_spks = egs.size(0)
        sisnr_mat = torch.stack(
            [torch.stack([self.sisnr(output_audios[s], egs[t]) for t in range(num_spks)]) for s in range(num_spks)]
        )
        max_sisnr, _ = torch.max(sisnr_mat.sum(dim=0), dim=0)
        return {"loss": -max_sisnr.mean()}
