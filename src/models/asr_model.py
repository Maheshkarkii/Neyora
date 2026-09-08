import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from src.models.cnn_extractor import CNNExtractor

class NepaliASR(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        cnn_out_channels: int = 64,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 2,
        rnn_dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super(NepaliASR, self).__init__()
        self.vocab_size = vocab_size
        self.n_mels = n_mels
        self.rnn_hidden_size = rnn_hidden_size
        self.bidirectional = bidirectional

        self.cnn_extractor = CNNExtractor(in_channels=1, n_mels=n_mels, cnn_out_channels=cnn_out_channels)
        cnn_feature_dim = self.cnn_extractor.out_dim

        self.rnn = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )

        rnn_out_dim = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size
        self.fc = nn.Linear(rnn_out_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        mel_specs: torch.Tensor,
        input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features, sub_lengths = self.cnn_extractor(mel_specs, input_lengths)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            features,
            sub_lengths.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, _ = self.rnn(packed_input)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        logits = self.fc(rnn_out)
        log_probs = self.log_softmax(logits)
        return log_probs, sub_lengths
