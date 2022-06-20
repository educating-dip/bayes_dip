import torch 

def list_norm_layers(model):

    """ compute list of names of all GroupNorm (or BatchNorm2d) layers in the model """
    norm_layers = []
    for (name, module) in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, torch.nn.GroupNorm) or isinstance(module,
                torch.nn.BatchNorm2d) or isinstance(module, torch.nn.InstanceNorm2d):
            norm_layers.append(name + '.weight')
            norm_layers.append(name + '.bias')
    return norm_layers

def count_parameters(model, norm_layers, include_biases):
    len_w = 0
    for name, param in model.named_parameters():
        name = name.replace('module.', '')
        if 'weight' in name and name not in norm_layers:
            len_w += param.data.numel()
        if include_biases and 'bias' in name and name not in norm_layers:
            len_w += param.data.numel()
    return len_w