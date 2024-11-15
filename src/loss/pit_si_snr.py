import torch
from torch import nn
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

class PITSiSNRLoss(nn.Module):

    def __init__(self):
        super().__init__()
        # self.si_sdr = ScaleInvariantSignalDistortionRatio() was
        self.si_sdr = ScaleInvariantSignalNoiseRatio()

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
        num_spks, B, L = egs.shape
        assert num_spks == 2, 'Unfortunately, we can only work with 2 speakers'

        first_score = self.si_sdr(output_audios, egs)
        second_score = self.si_sdr(torch.flip(output_audios, [0]), egs)
        print(f'dif: {first_score - second_score}')
        
        return {"loss": -torch.max(first_score, second_score)}
