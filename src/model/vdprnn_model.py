from src.model.dprnn_model import DPRNNModel
import torch.nn as nn
import torch

class VideoEncoder(nn.Module):
    def __init__(self, audio_len, video_dim, hidden_dim1, hidden_dim2, audio_dim, kernel_size, norm_type, **kwargs):
        super().__init__()
        self.upsample = nn.Upsample(audio_len)
        self.conv1 = nn.Conv1d(in_channels=video_dim, out_channels=hidden_dim1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim1, out_channels=hidden_dim2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv3 = nn.Conv1d(in_channels=hidden_dim2, out_channels=audio_dim, kernel_size=kernel_size, padding=kernel_size // 2)

        # nn.init.constant_(self.conv3.weight, 0)
        # nn.init.constant_(self.conv3.bias, 0)

        self.norm = getattr(nn, norm_type)(normalized_shape=audio_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: torch.Tensor [B, Frames, D_video], F - frames
        Returns:
            out: torch.Tensor [B, N, L]
        """
        out = self.upsample(x.permute(0, 2, 1))  # [B, D_video, L]
        # print('out.shape (must change only last dim):', out.shape)
        out = self.conv1(out)  # [B, hidden_dim, L]
        out = self.relu(out)  # [B, hidden_dim1, L]
        out = self.conv2(out)  # [B, N, L]
        out = self.relu(out)  # [B, hidden_dim2, L]
        out = self.conv3(out)  # [B, N, L]
        out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1)  # [B, N, L] 
        return out


class VDPRNNModel(nn.Module):
    def __init__(self, dprnn_config, video_encoder_config, pretrained_dprnn_path, **kwargs):
        super().__init__()
        self.video_encoder = VideoEncoder(**video_encoder_config)
        self.dprnn_model = DPRNNModel(**dprnn_config)
        if pretrained_dprnn_path:
            checkpoint = torch.load(pretrained_dprnn_path)
            self.dprnn_model.load_state_dict(checkpoint["state_dict"])


    def forward(self, mix: torch.Tensor, vid1: torch.Tensor, vid2: torch.Tensor, **batch) -> dict[list[torch.Tensor]]:
        encoded_vid1 = self.video_encoder(vid1)
        encoded_vid2 = self.video_encoder(vid2)

        conv1_param = [torch.norm(p).item() for p in self.video_encoder.conv1.parameters()]
        conv1_param = sum(conv1_param) / len(conv1_param)
        conv2_param = [torch.norm(p).item() for p in self.video_encoder.conv2.parameters()]
        conv2_param = sum(conv2_param) / len(conv2_param)
        conv3_param = [torch.norm(p).item() for p in self.video_encoder.conv3.parameters()]
        conv3_param = sum(conv3_param) / len(conv3_param)
        # norm_param = [torch.norm(p).item() for p in self.video_encoder.norm..parameters()]
        # norm_param = sum(norm_param) / len(norm_param)
        # print(f'encoder_norm (c1, c2, c3, norm):', conv1_param, conv2_param, conv3_param, torch.norm(self.video_encoder.norm.weight).item())

        return self.dprnn_model(mix, encoded_vid1, encoded_vid2)
    

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info