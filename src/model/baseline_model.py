from torch import nn
from torch.nn import Sequential


class BaselineModel(nn.Module):
    """
    Simple MLP
    """

    def __init__(self, n_feats, n_class, fc_hidden=512):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.net = Sequential(
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_feats),
            nn.ReLU(),
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, mix, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        print(mix.shape)
        mix = mix.unsqueeze(1)  # Преобразуем из (B, L) в (B, 1, L)
        mix = self.net(mix)
        mix = mix.permute(1, 0, 2)  # Из (B, SC, L) делаем (SC, B, L)
        # print(mix.shape)

        return {"outputs": mix}

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
