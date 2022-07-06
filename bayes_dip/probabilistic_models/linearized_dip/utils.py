
def get_inds_from_ordered_params(nn_model, ordered_nn_params):
    params = list(nn_model.parameters())

    inds_in_full_model = []
    for param in ordered_nn_params:
        inds_in_full_model.append(
                next(i for i, p in enumerate(params) if p is param))

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
