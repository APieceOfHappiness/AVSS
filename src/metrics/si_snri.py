import torch
from src.metrics.base_metric import BaseMetric
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

class SI_SNRiMetric(BaseMetric):
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

    def si_snr_i(self, preds: torch.Tensor, refs: torch.Tensor, mix: torch.Tensor):
        """
        Calculate SI-SNR improvement (SI-SNRi) for separated signals.
        Args:
            preds (Tensor): Predicted separated signals, tensor (num_spks, B, L)
            refs (Tensor): Reference signals, tensor (num_spks, B, L)
            mix (Tensor): Mixed signal, tensor (B, L)
        Returns:
            SI-SNRi score (float)
        """

        num_spks, B, L = refs.shape
        assert num_spks == 2, 'Unfortunately, we can only work with 2 speakers'

        first_score = self.si_sdr(preds, refs)
        second_score = self.si_sdr(torch.flip(preds, [0]), refs)
        mix_sisnr = self.si_sdr(torch.stack([mix, mix], dim=0), refs)
 
        return torch.max(first_score, second_score) - mix_sisnr

    def __call__(self, output_audios: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor, mix: torch.Tensor, **kwargs):
        """
        Metric calculation logic.
        
        Args:
            output_audios (Tensor): model output predictions, tensor of shape (num_spks, B, L)
            refs (Tensor): ground-truth signals, tensor of shape (num_spks, B, L)
            mix (Tensor): mixed signal, tensor of shape (B, L)
        Returns:
            metric (float): calculated SI-SNRi
        """
        refs = torch.stack([s1, s2])
        output_audios, refs, mix = output_audios.to(self.device), refs.to(self.device), mix.to(self.device)
        return self.si_snr_i(output_audios, refs, mix)
