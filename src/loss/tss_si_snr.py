import torch
from torch import nn
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

class TSSSiSNRLoss(nn.Module):

    def __init__(self):
        super().__init__()
        # self.si_sdr = ScaleInvariantSignalDistortionRatio() was
        self.si_sdr = ScaleInvariantSignalNoiseRatio()

    def forward(self, output_audio: torch.Tensor, s1: torch.Tensor, **batch):
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

        score = self.si_sdr(output_audio, s1)
        return {"loss": -score}
