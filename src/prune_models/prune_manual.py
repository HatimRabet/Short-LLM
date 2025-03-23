import torch


def prune_layers(model, start_layer, end_layer):
    # Example assumes transformer layers are accessible via model.model.layers
    pruned_layers = torch.nn.ModuleList(
        [layer for idx, layer in enumerate(model.model.layers) 
         if idx < start_layer or idx >= end_layer]
    )
    model.model.layers = pruned_layers
    model.config.num_hidden_layers = len(pruned_layers)
    return model