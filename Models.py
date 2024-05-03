import torch
from torch import nn


class Model_1(nn.Module):
    def __init__(self, 
                 static_channels: int, 
                 dynamic_history_channels: int, 
                 dynamic_future_channels: int, 
                 d_model: int, 
                 layers: int, 
                 dropout: float, 
                 out_channels: int, 
                 tgt_len: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=static_channels + dynamic_history_channels, 
            hidden_size=d_model, 
            num_layers=layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.decoder = nn.LSTM(
            input_size=static_channels + dynamic_future_channels, 
            hidden_size=d_model, 
            num_layers=layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.projection = nn.Linear(d_model, out_channels)
        self.tgt_len = tgt_len

    def forward(self, static_features: torch.Tensor, 
                history_features: torch.Tensor, future_features: torch.Tensor) -> torch.Tensor:
        """
        static_features shape: (batch_size, static_channels)
        history_features shape: (batch_size, seq_len, history_channels)
        future_features shape: (batch_size, pred_len, future_channels)
        """
        batch_size, seq_len, _ = history_features.shape
        _, pred_len, _ = future_features.shape
        encoder_in = torch.cat([history_features, static_features.unsqueeze(1).repeat(1, seq_len, 1)], dim=2)
        enc_out, enc_state = self.encoder(encoder_in)
        dec_in = torch.cat([future_features, static_features.unsqueeze(1).repeat(1, pred_len, 1)], dim=2)
        dec_out, _ = self.decoder(dec_in, enc_state)
        out = torch.cat([enc_out[:, -1:, :], dec_out], dim=1)
        return self.projection(out)[:, -self.tgt_len:, :]
        

class Model_2(nn.Module):
    def __init__(self, 
                 static_channels: int, 
                 history_channels: int, 
                 future_channels: int, 
                 out_channels: int, 
                 dropout: float, 
                 enc_channels: int, 
                 dec_channels: int,  
                 enc_layers: int, 
                 dec_layers: int, 
                 tgt_len: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=static_channels+history_channels, 
            hidden_size=enc_channels, 
            num_layers=enc_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.decoder = nn.LSTM(
            input_size=static_channels+future_channels+enc_channels, 
            hidden_size=dec_channels, 
            num_layers=dec_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.projection = nn.Linear(dec_channels, out_channels)
        self.tgt_len = tgt_len

    def forward(self, 
                static_features: torch.Tensor, 
                history_features: torch.Tensor, 
                future_features: torch.Tensor) -> torch.Tensor:
        """
        static_features shape: (batch_size, static_channels)
        history_features shape: (batch_size, seq_len, history_channels)
        future_features shape: (batch_size, pred_len, future_channels)
        """
        _, seq_len, _ = history_features.shape
        _, pred_len, _  = future_features.shape
        assert seq_len == pred_len
        enc_in = torch.cat([static_features.unsqueeze(1).repeat(1, seq_len, 1), history_features], dim=2)
        enc_out, _ = self.encoder(enc_in)
        dec_in = torch.cat([enc_out, static_features.unsqueeze(1).repeat(1, pred_len, 1), future_features], dim=2)
        dec_out, _ = self.decoder(dec_in)
        out = self.projection(dec_out)
        return out[:, -self.tgt_len:, :]
