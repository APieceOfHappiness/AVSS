import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: [B, for_conv, for_norm] <-> [B, emb_dim, len]

        Returns:
          x: [B, for_conv, for_norm]
        """
        out = self.conv(x)
        out = self.relu(out)
        out = self.layer_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        return out


class AudioEncoder(nn.Module):
    def __init__(self, audio_emb_dim_in, audio_emb_dim_out, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = ConvBlock(in_channels=audio_emb_dim_in,
                              out_channels=audio_emb_dim_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: [B, audio_dim, audio_len]
        Returns:
          out: [B, audio_dim, audio_len]
        """
        out = self.conv(x.unsqueeze(1))
        return out

class AudioDecoder(nn.Module):
    def __init__(self, audio_emb_dim_in, audio_emb_dim_out, kernel_size, stride, padding=0):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_channels=audio_emb_dim_in,
                                                 out_channels=audio_emb_dim_out,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 output_padding=stride - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: [B, audio_dim, audio_len]
        Returns:
          out: [B, audio_dim, audio_len]
        """
        out = self.conv_transpose(x).squeeze(1)
        return out


class VideoEncoder(nn.Module):
    def __init__(self, video_emb_dim_in, video_emb_dim_out, upsample_size, kernel_size, stride, padding=0):
        super().__init__()
        self.upsample = nn.Upsample(upsample_size)
        self.conv = ConvBlock(in_channels=video_emb_dim_in,
                              out_channels=video_emb_dim_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding) # PADDING???

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: [B, video_dim, video_len]
        Returns:
          out: [B, video_dim, video_len]
        """
        out = self.upsample(x)
        out = self.conv(out)
        return out


class AudioSub(nn.Module):
    def __init__(self, hidden_dim, audio_len, num_layers=1, kernel_size=3):
        super().__init__()

        assert kernel_size == 3, 'Currently kernel_size != 3 is not available, due to stride'
        num_layers -= 1  # Пользователь указывает кол-во слоёв, которые он хочет видеть (Чтобы получить 3 разные матрицы, нужно сжать 2 раза, отсюда и -1)
        # assert audio_len % (2 ** num_layers) == 0, "audio_len must be divided by 2 ** num_layers" 

        self.main_down_sample = nn.ModuleList([
              ConvBlock(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=2, padding=(kernel_size - 1) // 2)
              for _ in range(num_layers)
        ])
        self.post_up_sample = nn.functional.interpolate
        # self.post_up_sample = nn.ModuleList([
        #    nn.Upsample(audio_len // (2 ** i) + int(audio_len % (2 ** i) != 0))
        #    for i in range(num_layers)
        # ]) 
        self.pre_down_sample = nn.ModuleList([
              ConvBlock(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=2, padding=(kernel_size - 1) // 2)
              for _ in range(num_layers)
        ])
        self.micro_compressor = nn.ModuleList([
              ConvBlock(in_channels=hidden_dim + (hidden_dim if i > 0 else 0) + (hidden_dim if i < num_layers else 0), 
                        out_channels=hidden_dim, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
              for i in range(num_layers + 1)
        ])  # сжимает 3 соседние вершины
        self.macro_compressor = ConvBlock(in_channels=hidden_dim * (num_layers + 1), 
                                          out_channels=hidden_dim, 
                                          kernel_size=1, stride=1, 
                                          padding=0)  # Сжимает все вершины перед подачей в fusion

        # self.final_up_sample = nn.Upsample(audio_len)  # upsample не имеет обучаемых параметров
        self.final_up_sample = nn.functional.interpolate

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: [B, N, L]
        Returns:
          out: [B, N, L]
        """
        saved_tensors = [x]
        for i in range(self.num_layers):
            new_tensor = self.main_down_sample[i](saved_tensors[-1])  # [B, N, T_i]
            saved_tensors.append(new_tensor)

        compressed_tensors = []
        for i in range(self.num_layers + 1):
            pre_tensor = None
            post_tensor = None
            if i > 0:
                pre_tensor = self.pre_down_sample[i - 1](saved_tensors[i - 1])  # [B, N, T_i]
            if i < self.num_layers:
                # post_tensor = self.post_up_sample[i](saved_tensors[i + 1])  # [B, N, T_i]
                post_tensor = self.post_up_sample(saved_tensors[i + 1], size=saved_tensors[i].shape[-1])

            compressed_tensors.append(torch.cat(
                ([post_tensor] if post_tensor is not None else []) + \
                ([saved_tensors[i]]) + \
                ([pre_tensor] if pre_tensor is not None else []),
            dim=1))  # [B, 3 * N, T_i]

            compressed_tensors[i] = self.micro_compressor[i](compressed_tensors[i])  # [B, N, T_i]

        # saved_tensors = torch.cat([self.final_up_sample(saved_tensor) for saved_tensor in saved_tensors], dim=1)  # [B, N * num_layers, T_i]
        saved_tensors = torch.cat([
            self.final_up_sample(saved_tensor, size=saved_tensors[0].shape[-1]) 
            for saved_tensor in saved_tensors
        ], dim=1)  # [B, N * num_layers, T_i]
        return self.macro_compressor(saved_tensors)


VideoSub = AudioSub  # video sub ничем не отличается от аудио

class ThalamicSub(nn.Module):
    def __init__(self, audio_dim, video_dim):
        super().__init__()
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.linear_audio = nn.Linear(in_features=audio_dim, out_features=audio_dim)
        self.linear_video = nn.Linear(in_features=video_dim, out_features=video_dim)


    def forward(self, audio_emb: torch.Tensor, video_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
          audio_emb: [B, N, audio_len]
          video_emb: [B, C, video_len]

        Returns:
          audio_emb: [B, N, audio_len]
          video_emb: [B, C, video_len]
        """

        scaled_video = nn.functional.interpolate(video_emb.unsqueeze(0), size=audio_emb.shape[-2:]).squeeze(0)
        scaled_audio = nn.functional.interpolate(audio_emb.unsqueeze(0), size=video_emb.shape[-2:]).squeeze(0)

        audio_emb = (audio_emb + scaled_video).permute(0, 2, 1)
        video_emb = (video_emb + scaled_audio).permute(0, 2, 1)

        audio_emb = self.linear_audio(audio_emb)
        video_emb = self.linear_video(video_emb)

        return audio_emb.permute(0, 2, 1), video_emb.permute(0, 2, 1)
    

class CTCNet(nn.Module):
    def __init__(self, audio_blocks_cnt, 
                 audio_emb_dim,
                 video_emb_dim,
                 video_blocks_cnt, 
                 audio_block_config, 
                 video_block_config, 
                 thalamic_block_config, 
                 audio_encoder_config, 
                 video_encoder_config,
                 **kwargs):
        super().__init__()
        self.audio_blocks_cnt = audio_blocks_cnt        
        self.video_blocks_cnt = video_blocks_cnt
        self.audio_encoder = AudioEncoder(audio_emb_dim_in=1, audio_emb_dim_out=audio_emb_dim, **audio_encoder_config)
        self.video_encoder = VideoEncoder(video_emb_dim_in=512, video_emb_dim_out=video_emb_dim, **video_encoder_config)
        self.audio_blocks = nn.ModuleList([
            AudioSub(**audio_block_config) for _ in range(audio_blocks_cnt)
        ])
        self.video_blocks = nn.ModuleList([
            VideoSub(**video_block_config) for _ in range(video_blocks_cnt)
        ])
        self.thalamic_blocks = nn.ModuleList([
            ThalamicSub(**thalamic_block_config) for _ in range(video_blocks_cnt)
        ])

        self.last_linear = nn.Linear(in_features=audio_emb_dim, out_features=audio_emb_dim)
        self.last_relu = nn.ReLU()

        self.decoder = AudioDecoder(audio_emb_dim_in=audio_emb_dim, audio_emb_dim_out=1, **audio_encoder_config)

    def forward(self, mix: torch.Tensor, vid1: torch.Tensor, **batch) -> torch.Tensor:
        """
        Args:
          mix: [B, audio_len]
          vid1: [B, video_len, video_dim]

        Returns:
          out: [B, audio_len]
        """
        vid1 = vid1.permute(0, 2, 1)

        audio_embs = self.audio_encoder(mix)
        video_embs = self.video_encoder(vid1)

        saved_audio_embs = audio_embs
        audio_res = audio_embs * 0
        video_res = video_embs * 0

        for i in range(self.video_blocks_cnt):
            audio_embs = self.audio_blocks[i](audio_embs + audio_res)
            video_embs = self.video_blocks[i](video_embs + video_res)
            
            audio_res = audio_embs
            video_res = video_embs

            audio_embs, video_embs = self.thalamic_blocks[i](audio_embs, video_embs)

        for i in range(self.video_blocks_cnt, self.audio_blocks_cnt):
            audio_embs = self.audio_blocks[i](audio_embs)  # [B, audio_dim, audio_len]

        audio_embs = self.last_linear(audio_embs.permute(0, 2, 1)).permute(0, 2, 1)  # [B, audio_dim, audio_len]
        mask = self.last_relu(audio_embs)

        return {'output_audio': self.decoder(saved_audio_embs * mask)}
    
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