import torch 

def flatten_grad_functorch(inds_from_ordered_params, grads):
    jacs = []
    for ind in inds_from_ordered_params:
        jacs.append(grads[ind].detach().reshape(-1))

    jacs = torch.cat(jacs, dim=0)
    return jacs

def unflatten_nn_functorch(model, inds_from_ordered_params, slices_from_ordered_params, weights):
    weight_list = []
    slice_iter = iter(slices_from_ordered_params)
    for ind, param in enumerate(model.parameters()):
        if ind not in inds_from_ordered_params:
            weight_list.append(
                torch.zeros_like(param, dtype=torch.float).to(weights.device, non_blocking=True)
            )
        else:
            weight_list.append(weights[next(slice_iter)].view(*param.shape))

    # functorch.make_functional returns tuple(params)
    return tuple(weight_list)

def get_inds_from_ordered_params(model, ordered_nn_params):

    inds_in_full_model = []
    for ind, (_, param) in enumerate(model.named_parameters()):
        if any(param is ordered_nn_param for ordered_nn_param in ordered_nn_params):
            inds_in_full_model.append(ind)

    return inds_in_full_model

def get_slices_from_ordered_params(ordered_nn_params):

    slices = []
    w_pointer = 0
    for param in ordered_nn_params:
        slices.append(
            slice(w_pointer, w_pointer + param.data.numel())
            )
        w_pointer += param.data.numel()
    
    return slices
