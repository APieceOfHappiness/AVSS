import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from src.metrics.base_metric import BaseMetric
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

class STOIMetric(BaseMetric):
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
        self.stoi = ShortTimeObjectiveIntelligibility(fs=8000).to(self.device)

    def stoi_score(self, preds: torch.Tensor, refs: torch.Tensor):
        """
        Calculate SDR improvement (SDRi) for separated signals.
        Args:
            preds (Tensor): Predicted separated signals, tensor (num_spks, B, L)
            refs (Tensor): Reference signals, tensor (num_spks, B, L)
            mix (Tensor): Mixed signal, tensor (B, L)
        Returns:
            SDRi score (float)
        """

        num_spks, B, L = refs.shape
        assert num_spks == 2, 'Unfortunately, we can only work with 2 speakers'

        score = 0
        preds_flipped = torch.flip(preds, [0])
        for batch_idx in range(B):
            first_score = self.stoi(preds[:, batch_idx, :], refs[:, batch_idx, :])
            second_score = self.stoi(preds_flipped[:, batch_idx, :], refs[:, batch_idx, :])
            score += torch.max(first_score, second_score)

 
        return score / B

    def batch_peak_norm(self, pred_audio: torch.Tensor, ref_audio: torch.Tensor) -> torch.Tensor:
        """
        Normalize each predicted audio signal in the batch to match the peak amplitude 
        of the corresponding reference audio signal, handling multiple speakers.

        Args:
            pred_audio (torch.Tensor): Predicted audio signals, tensor of shape (num_spks, B, L)
            ref_audio (torch.Tensor): Reference audio signals (mix), tensor of shape (B, L)

        Returns:
            torch.Tensor: Normalized predicted audio signals, tensor of shape (num_spks, B, L)
        """
        # Compute peak amplitudes for each speaker and each batch
        pred_audio = pred_audio - torch.mean(pred_audio, dim=2, keepdim=True)
        peak_pred = torch.max(torch.abs(pred_audio), dim=2, keepdim=True)[0]
        peak_ref = torch.max(torch.abs(ref_audio), dim=1, keepdim=True)[0]
        
        epsilon = 1e-9
        normalized_audio = pred_audio * (peak_ref.unsqueeze(0) / (peak_pred + epsilon))
        return normalized_audio

    def __call__(self, output_audios: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor, mix: torch.Tensor, **kwargs):
        """
        Metric calculation logic.
        
        Args:
            output_audios (Tensor): model output predictions, tensor of shape (num_spks, B, L)
            refs (Tensor): ground-truth signals, tensor of shape (num_spks, B, L)
            mix (Tensor): mixed signal, tensor of shape (B, L)
        Returns:
            metric (float): calculated SDRi
        """
        refs = torch.stack([s1, s2])
        output_audios, refs, mix = output_audios.to(self.device), refs.to(self.device), mix.to(self.device)
        output_audios = self.batch_peak_norm(output_audios, mix)

        return self.stoi_score(output_audios, refs)
