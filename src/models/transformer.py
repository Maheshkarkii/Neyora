import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, List
from src.models.transformer_encoder import TransformerEncoder
from src.models.transformer_decoder import TransformerDecoder

class Transformer(nn.Module):
    """
    Complete Sequence-to-Sequence Transformer Model for Nepali-to-English Translation
    implemented entirely from scratch in PyTorch.
    
    Tensor Shapes:
    - `src`         : [B, S], Source Nepali token IDs
    - `tgt`         : [B, T], Target English token IDs
    - `outputs`     : [B, T, V_tgt], Output logits over target vocabulary
    - `enc_attns`   : List of [B, H, S, S] attention maps per encoder layer
    - `dec_self_attns`: List of [B, H, T, T] attention maps per decoder layer
    - `dec_cross_attns`: List of [B, H, T, S] attention maps per decoder layer
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        max_len: int = 5000,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.device = device

        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_len=max_len
        )

        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_len=max_len
        )

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    @staticmethod
    def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        Creates padding mask: [B, 1, 1, L] where True represents <PAD> positions.
        """
        return (seq == pad_idx).unsqueeze(1).unsqueeze(2) # [B, 1, 1, L]

    @staticmethod
    def create_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """
        Creates upper-triangular causal mask: [1, 1, size, size]
        where True represents masked future token positions.
        """
        mask = torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(1) # [1, 1, size, size]

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return self.create_padding_mask(src, self.pad_idx)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        # Combined padding mask and causal lookahead mask
        tgt_pad_mask = self.create_padding_mask(tgt, self.pad_idx) # [B, 1, 1, T]
        tgt_len = tgt.size(1)
        causal_mask = self.create_causal_mask(tgt_len, tgt.device) # [1, 1, T, T]
        return tgt_pad_mask | causal_mask # [B, 1, T, T]

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        return self.encoder(src, src_mask=src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        return self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        # Masks
        src_mask = self.make_src_mask(src) # [B, 1, 1, S]
        tgt_mask = self.make_tgt_mask(tgt) # [B, 1, T, T]
        memory_mask = self.make_src_mask(src) # [B, 1, 1, S]

        # 1. Encode Source: [B, S, D]
        memory, enc_attns = self.encoder(src, src_mask=src_mask)

        # 2. Decode Target: [B, T, D]
        dec_out, dec_self_attns, dec_cross_attns = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )

        # 3. Project to Target Vocabulary Logits: [B, T, V_tgt]
        logits = self.output_projection(dec_out)

        attentions = {
            "encoder_attentions": enc_attns,
            "decoder_self_attentions": dec_self_attns,
            "decoder_cross_attentions": dec_cross_attns
        }

        return logits, attentions
