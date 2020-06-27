import torch


def btdot(large, small):
    """Batch dot tensor product.

    If the dimension of small is `N+1`, perform a batch N-dimensional dot product.
        The dot operation produces the sum of the multiplied components. For example,
        if small is 2D, perform a vector-vector multiplication. The first dimension
        of both tensors is a batch dimension and it must match or be broadcastable.
        If large has extra dimensions, they are considered batch dimensions too. 

    Args:
        large (tensor): its last `N` dimensions are multiplied by small. The remaining
            dimensions are batch dimensions.
        small (tensor): a tensor whose first dimension is the batch, and the
            remaining ones are to be multiplied. Its dimension must be equal or
            saller than those of large.
    Returns:
        tensor: the result of the product.
    """
    dim_diff = large.dim() - small.dim()
    batch_dim = small.size(0)
    extra_dims = [1] * dim_diff
    remaining_dims = small.size()[1:]
    sview = small.view(batch_dim, *extra_dims, *remaining_dims)
    return (large * sview).sum(tuple(range(dim_diff + 1, large.dim())))


def assemble(length, conn, inputs):
    '''Assemble the inputs according to the connectivity.'''
    storage = torch.zeros(length, inputs.size(2))
    for j in range(storage.size(1)):
        for i in range(conn.size(1)):
            # Add the elements of the input vector, according to the indices in
            #  conn.
            # Dev notes: This is faster but equivalent to
            # storage[:, j].put_(conn[:, i], inputs[:, i, j], accumulate=True)
            # conn and inputs have two batch dimensions. Instead of the loop over
            # i we could have generated views of both tensors.
            storage[:, j] += torch.bincount(
                conn[:, i], weights=inputs[:, i, j], minlength=storage.size(0)
            )
    return storage
