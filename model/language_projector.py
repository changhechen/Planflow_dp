import torch
import torch.nn as nn
from typing import Optional, Tuple


class LanguageProjector(nn.Module):
    def __init__(
        self,
        in_dim: int = 3584,
        proj_dim: int = 16,
        num_tokens: int = 32,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        self.num_tokens = num_tokens

        self.proj = nn.Linear(in_dim, proj_dim)

    def forward(
        self,
        lang_tokens: torch.Tensor,      # (B, T, N, D) or (T, N, D)
        lang_mask: Optional[torch.Tensor] = None,  # (B, T, N) or (T, N)
    ):
        #print(lang_mask)
        orig_ndim = lang_tokens.ndim
        if orig_ndim == 3:
            # (T, N, D) -> (1, T, N, D)
            lang_tokens = lang_tokens.unsqueeze(0)
            if lang_mask is not None and lang_mask.ndim == 2:
                lang_mask = lang_mask.unsqueeze(0)

        B, T, N, D = lang_tokens.shape
        assert N == self.num_tokens
        assert D == self.in_dim

        projected = self.proj(lang_tokens)          # (B, T, N, proj_dim)

        mask_flat = None
        step_has_valid = None

        if lang_mask is not None:
            # lang_mask: 1 = valid token, 0 = padding
            lang_mask_bool = lang_mask.to(torch.bool)           # (B, T, N)

            # Zero out invalid tokens
            projected = projected * lang_mask_bool.unsqueeze(-1).to(projected.dtype)

            # flat mask aligned with flattened features
            mask_full = lang_mask_bool.unsqueeze(-1).expand(-1, -1, -1, self.proj_dim)
            mask_flat = mask_full.reshape(B, T, N * self.proj_dim)  # (B, T, N*proj_dim)

            # Time-step validity: True if at least one token is valid at that step
            step_has_valid = lang_mask_bool.any(dim=-1)          # (B, T)

        projected_flat = projected.reshape(B, T, N * self.proj_dim)  # (B, T, N*proj_dim)

        if orig_ndim == 3:
            projected_flat = projected_flat[0]
            if mask_flat is not None:
                mask_flat = mask_flat[0]
            if step_has_valid is not None:
                step_has_valid = step_has_valid[0]

        # Return both the flattened features and step-level mask
        # torch.set_printoptions(profile="full")
        # print(mask_flat)
        # print(step_has_valid)
        return projected_flat, mask_flat, step_has_valid
