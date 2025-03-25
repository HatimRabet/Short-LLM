import torch.nn as nn


def prune_layers(model, start_layer, end_layer):
    
    # Keep only layers outside the pruning range
    pruned_layers = nn.ModuleList(
        [layer for idx, layer in enumerate(model.model.layers) 
         if idx <= start_layer or idx > end_layer]
    )
    # Assign back to the model
    model.model.layers = pruned_layers
    model.config.num_hidden_layers = len(pruned_layers)

    return model