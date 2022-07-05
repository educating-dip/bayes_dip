import torch

def flatten_grad_functorch(inds_from_ordered_params, grads):
    jacs = []
    for ind in inds_from_ordered_params:
        jacs.append(grads[ind].detach().reshape(-1))

    jacs = torch.cat(jacs, dim=0)
    return jacs

def unflatten_nn_functorch(model, inds_from_ordered_params, slices_from_ordered_params, weights):
    params = list(model.parameters())
    weight_list = [None] * len(params)

    for ind, slice_param in zip(inds_from_ordered_params, slices_from_ordered_params):
        weight_list[ind] = weights[slice_param].view(
                *params[ind].shape)

    for ind, weight in enumerate(weight_list):
        if weight is None:
            weight_list[ind] = torch.zeros(
                    params[ind].shape, dtype=torch.float, device=weights.device)

    # functorch.make_functional returns tuple(params)
    return tuple(weight_list)
