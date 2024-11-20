import torch
from torch import nn
import torch.nn.functional as F


class DPRNNBlock(nn.Module):
    def __init__(self, feature_dim, hidden_channels, rnn_type='LSTM', norm_type='LayerNorm', dropout=0, bidirectional=True):
        super().__init__()
        self.inter_chunk_rnn = getattr(nn, rnn_type)(input_size=feature_dim, hidden_size=hidden_channels, 
                                                     batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.intra_chunk_rnn = getattr(nn, rnn_type)(input_size=feature_dim, hidden_size=hidden_channels, 
                                                     batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        self.inter_chunk_norm = getattr(nn, norm_type)(normalized_shape=feature_dim)  # TO CHECK
        self.intra_chunk_norm = getattr(nn, norm_type)(normalized_shape=feature_dim)  # TO CHECK

        self.intra_chunk_linear = nn.Linear(
            in_features=hidden_channels * 2 if bidirectional else hidden_channels,
            out_features=feature_dim
        )
        self.inter_chunk_linear = nn.Linear(
            in_features=hidden_channels * 2 if bidirectional else hidden_channels,
            out_features=feature_dim
        )

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        """
        Model forward method.

        Args:
            x (Tensor): [B, N, K, S]
        Returns:
            out (Tensor): [B, N, K, S]
        """
        B, N, K, S = mix.shape  # [B, N, K, S]

        # intra
        intra_out = mix.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)  # [B * S, K, N]
        intra_out, _ = self.intra_chunk_rnn(intra_out)  # [B * S, K, N]
        intra_out = self.intra_chunk_linear(intra_out)  # [B * S, K, N]
        intra_out = self.intra_chunk_norm(intra_out.view(B, S, K, N))  # B, S, K, N
        intra_out = intra_out.permute(0, 3, 2, 1) + mix  # [B, N, K, S]

        # inter
        inter_out = intra_out.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)  # [B * K, S, N]
        inter_out, _ = self.inter_chunk_rnn(inter_out)  # [B * K, S, N]
        inter_out = self.inter_chunk_linear(inter_out)  # [B * K, S, N]
        inter_out = self.inter_chunk_norm(inter_out.view(B, K, S, N))  # B, K, S, N
        inter_out = inter_out.permute(0, 3, 1, 2) + intra_out  # [B, N, K, S]

        return inter_out


class Gate(nn.Module):
    def __init__(self, hidden_dim: torch.Tensor) -> torch.Tensor:
        super().__init__()
        self.data_extrator = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, 1))  # I deleted TanH
        self.importance_extractor = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.data_extrator(x) * self.importance_extractor(x)


class DPRNNModel(nn.Module):
    def __init__(self, feature_dim, num_of_speakers, num_dprnn_layers, norm_type,
                 feature_extractor_config, split_config, dprnn_block_config, audio_only, **kwargs):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()
        self.audio_only = audio_only
        self.feature_dim = feature_dim
        self.num_of_speakers = num_of_speakers
        self.split_config = split_config
        self.relu = nn.ReLU()

        self.feature_extractor = nn.Conv1d(in_channels=1, out_channels=feature_dim, **feature_extractor_config)
        
        self.norm = getattr(nn, norm_type)(normalized_shape=feature_dim)
        self.dprnn_blocks = nn.Sequential(*[
            DPRNNBlock(**dprnn_block_config) for _ in range(num_dprnn_layers)
        ])
        self.gate = Gate(feature_dim)

        self.emb_expansion = nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim * num_of_speakers, kernel_size=1)

        self.decoder = nn.ConvTranspose1d(in_channels=feature_dim, out_channels=1, output_padding=1, **feature_extractor_config)



    def _split_input(self, mix: torch.Tensor, chunks_length: int, hop_size: int) -> torch.Tensor:
        """
        transform every sample to a 3D tensor

        Args:
            mix (Tensor): 3D tensor [B, N, L]
            chunks_length (int): length of one chunks (K)
            hop_size (int): overlap period (P)
        Returns:
            output (Tensor): 4D tensor [B, N, K, S]
        """
        assert mix.shape[-1] % chunks_length == 0, 'input.shape[-1] must be divided by chunks_length'
        assert chunks_length % hop_size == 0, 'chunks_length must be divided by hop_size'

        pad = chunks_length - hop_size

        mix_padded = F.pad(mix, (pad, pad), "constant", 0)
        chunk_cnt = mix.shape[-1] // hop_size + (chunks_length // hop_size - 1)
        chunks = []
        for chunk_id in range(chunk_cnt):
            chunk = mix_padded[:, :, hop_size * chunk_id: hop_size * chunk_id + chunks_length]
            chunks.append(chunk)

        return torch.stack(chunks, dim=-1)  # Что по CUDA памяти?

    def _overlap_add(self, mix: torch.Tensor, chunks_length: int, hop_size: int) -> torch.Tensor:
        """
        Transform every sample to the original length.

        Args:
            mix (Tensor): 5D tensor [num_of_speakers, B, N, K, S]
        Returns:
            output (Tensor): 4D tensor [num_of_speakers, B, N, L]
        """
        num_of_speakers, B, N, K, S = mix.shape
        pad = chunks_length - hop_size
        L = hop_size * S + pad

        batch_size = num_of_speakers * B * N
        mix_flat = mix.reshape(batch_size, K, S)

        chunk_ids = torch.arange(S, device=mix.device)
        idx_base = hop_size * chunk_ids
        indices = idx_base.unsqueeze(1) + torch.arange(K, device=mix.device).unsqueeze(0)
        indices = indices.view(1, -1).repeat(batch_size, 1)

        mix_flat = mix_flat.permute(0, 2, 1).contiguous().view(batch_size, -1)

        reduced_tensor_flat = torch.zeros((batch_size, L), device=mix.device)
        reduced_tensor_flat.scatter_add_(1, indices, mix_flat)

        reduced_tensor = reduced_tensor_flat.view(num_of_speakers, B, N, L)
        reduced_tensor = reduced_tensor[:, :, :, pad:-pad]
        return reduced_tensor


    def forward(self, mix: torch.Tensor, vid1: torch.Tensor = None, vid2: torch.Tensor = None, **batch) -> dict[list[torch.Tensor]]:
        """
        Model forward method.

        Args:
            mix (Tensor): input mix audios.
        Returns:
            output (dict[list[torch.Tensor]]): output dict containing separated audios.
        """
        B, L = mix.shape
        mix = mix.unsqueeze(1)  # [B, 1, L]

        mix = self.feature_extractor(mix)  # [B, N, L]
        mask = self.norm(mix.permute(0, 2, 1)).permute(0, 2, 1)  # [B, N, L] (!)

        if vid1 is not None and vid2 is not None and not self.audio_only:
            mask += vid1 + vid2

        mask = self._split_input(mix=mask, **self.split_config)  # [B, N, K, S]

        #  for stability
        mask = self.dprnn_blocks(mask)  # [B, N, K, S]
        mask = self.relu(mask)  # [B, N, K, S] (!)
        
        # ts
        B, N, K, S = mask.shape
        mask = self.emb_expansion(mask)  # [B, N * num_of_speakers, K, S]


        mask = mask.view(B, N, self.num_of_speakers, K, S).permute(2, 0, 1, 3, 4)  # [num_of_speakers, B, N, K, S]
        mask = self._overlap_add(mix=mask, **self.split_config)  # [num_of_speakers, B, N, L]

        gated_masks = [self.gate(speaker_mask) for speaker_mask in mask]  # [num_of_speakers, B, N, L]
        gated_masks = torch.stack(gated_masks)  # [num_of_speakers, B, N, L]

        gated_masks = self.relu(gated_masks)  # [num_of_speakers, B, N, L] WOOOW

        output_audios = torch.stack([self.decoder(gated_masks[speaker_id] * mix)
                                     for speaker_id in range(self.num_of_speakers)]).squeeze(2)

        return {"output_audios": output_audios}

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