import torch
from src.metrics.base_metric import BaseMetric
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

class TSSSiSNRiMetric(BaseMetric):
    def __init__(self, device="auto", *args, **kwargs):
        """
        SI-SNRi metric class for source separation task. Calculates the
        Scale-Invariant Signal-to-Noise Ratio improvement (SI-SNRi)
        between predicted and reference signals, given a mixed signal.

        Args:
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.si_sdr = ScaleInvariantSignalNoiseRatio().to(self.device)

    def si_snr_i(self, output_audio: torch.Tensor, mix: torch.Tensor, s1: torch.Tensor):
        """
        Calculate SI-SNR improvement (SI-SNRi) for separated signals.
        Args:
            output_audio (Tensor): Predicted separated signals, tensor (B, L)
            s1 (Tensor): Reference signals, tensor (B, L)
            mix (Tensor): Mixed signal, tensor (B, L)
        Returns:
            SI-SNRi score (float)
        """

        score = self.si_sdr(output_audio, s1)
        mix_score = self.si_sdr(mix, s1)

        return score - mix_score

    def __call__(self, output_audio: torch.Tensor, mix: torch.Tensor, s1: torch.Tensor, **kwargs):
        """
        Metric calculation logic.
        
        Args:
            output_audios (Tensor): model output predictions, tensor of shape (num_spks, B, L)
            refs (Tensor): ground-truth signals, tensor of shape (num_spks, B, L)
            mix (Tensor): mixed signal, tensor of shape (B, L)
        Returns:
            metric (float): calculated SI-SNRi
        """
        output_audio, mix, s1 = output_audio.to(self.device), mix.to(self.device), s1.to(self.device)
        return self.si_snr_i(output_audio, mix, s1)
