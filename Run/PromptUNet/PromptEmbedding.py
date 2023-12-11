import torch
from torch import nn
from typing import Any, Optional, Tuple, Type
import numpy as np
class PositionEmbeddingRandom(nn.Module):

    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class PromptEmbedder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        input_image_size: Tuple[int, int],device
    ) -> None:

        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size

        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2) # for every b_i vector, a cos(b_i^Tx) and sin(b_i^Tx) output is created therefore half embed_dim

        self.num_point_embeddings: int = 3  # cup disc and background classes
        point_embeddings = [nn.Embedding(1, embed_dim,device=device) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim) # for padding class


    def forward(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)

        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        # point_embedding[
        #     labels == -1] = 0.0  # set padded point embeddings to 0 (have been put thru point_embedding func)
        # point_embedding[
        #     labels == -1] += self.not_a_point_embed.weight  # set padded point embeddings to not a point signifier


        point_embedding[labels.squeeze(-1) == 0] += self.point_embeddings[0].weight
        point_embedding[labels.squeeze(-1) == 1] += self.point_embeddings[1].weight
        point_embedding[labels.squeeze(-1) == 2] += self.point_embeddings[2].weight
        return point_embedding



