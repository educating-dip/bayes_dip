import torch 

def flatten_grad_functorch(model, norm_layers, grads, include_biases=False):
    jacs = []
    for (name, _), grad in zip(model.named_parameters(), grads):
        name = name.replace('module.', '')
        if "weight" in name and name not in norm_layers:
            jacs.append(grad.detach().reshape(-1))

        if include_biases and 'bias' in name and name not in norm_layers:
            jacs.append(grad.detach().flatten())

    jacs = torch.cat(jacs, dim=0)
    return jacs

def unflatten_nn_functorch(model, norm_layers, weights, include_biases=False):
    w_pointer = 0
    weight_list = []
    for name, param in model.named_parameters():
        name = name.replace('module.', '')
        if 'weight' in name and name not in norm_layers:
            len_w = param.data.numel()
            weight_list.append(weights[w_pointer:w_pointer +len_w].view(param.shape).float().to(weights.device, non_blocking=True))
            w_pointer += len_w
        if include_biases and 'bias' in name and name not in norm_layers:
            len_w = param.data.numel()
            weight_list.append(weights[w_pointer:w_pointer + 
                                       len_w].view(param.shape).float().to(weights.device, non_blocking=True))
            w_pointer += len_w
        
        # Append zero batchnorm parameters, so that unflattened weights have
        # exactly same shape as func_model params, but ensure that batchnorm
        # parameters don't affect JvPs or vJPs.
        if name in norm_layers or (not include_biases and 'bias' in name and name not in norm_layers):
            len_w = param.data.numel()
            weight_list.append(torch.zeros_like(param).float().to(weights.device, non_blocking=True))

    # functorch.make_functional returns tuple(params)
    return tuple(weight_list)