
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
