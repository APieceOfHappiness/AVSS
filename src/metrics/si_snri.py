import torch
from src.metrics.base_metric import BaseMetric

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
        num_spks = refs.size(0)
        
        mix_sisnr = torch.stack([self.sisnr(mix, refs[i]) for i in range(num_spks)], dim=0)

        sisnr_mat = torch.stack(
            [torch.stack([self.sisnr(preds[s], refs[t]) for t in range(num_spks)], dim=0) for s in range(num_spks)],
            dim=0
        )  # shape: (num_spks, num_spks, B)

        max_sisnr, _ = torch.max(sisnr_mat.sum(dim=0), dim=0)
        return (max_sisnr - mix_sisnr.mean(dim=0)).mean()

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
