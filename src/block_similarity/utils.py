import torch
import torch.nn.functional as F

def angular_distance(x_l, x_l_plus_n) -> torch.Tensor:
    """Compute the angular distance between layer outputs"""
    cosine_similarity = F.cosine_similarity(x_l, x_l_plus_n, dim=1, eps=1e-8)
    return torch.acos(cosine_similarity.clamp(min=-1, max=1)) / torch.pi


def compute_block_distances(hidden_states, n_layers_to_skip):
    """Compute and return angular distances for each block of layers."""
    block_distances = []
    n_layers = len(hidden_states)

    for l in range(num_layers - n_layers_to_skip):
        block_dist = angular_distance(hidden_states[l], hidden_states[l + n_layers_to_skip]).mean().item()
        block_distances.append(block_distance)
    
    return block_distances


def get_last_non_padded_tokens(hidden_states, attention_mask) -> List[torch.Tensor]:
    """Get last non-padded tokens for each layer."""
    last_non_padded_hidden_states = []
    for layer in hidden_states:
        batch_size, _, _ = layer.size()
        batch_last_tokens = []
        for batch in range(batch_size):
            last_non_pad_index = attention_mask[batch].nonzero(as_tuple=True)[0].max()
            last_token = layer[batch, last_non_pad_index, :]
            batch_last_tokens.append(last_token.unsqueeze(0))
        last_non_padded_hidden_states.append(torch.cat(batch_last_tokens, dim=0))
    return last_non_padded_hidden_states