'''
Refer to https://oneapi-src.github.io/oneDNN/
'''
from typing import List, Tuple
from math import exp
dims_t = List[int]
tensor_t = List[float]

def _get_size(src_dims: dims_t, axis: int) -> Tuple[int, int, int]:
    # TODO: assert sdims

    outer_size = 1
    inner_size = 1
    
    for i in range(axis):
        outer_size *= src_dims[i]
    for i in range(axis + 1, len(src_dims)):
        inner_size *= src_dims[i]
    axis_size = src_dims[axis]
    
    return outer_size, inner_size, axis_size

def softmax(src: tensor_t, src_dims: dims_t, axis: int) -> tensor_t:
    outer_size, inner_size, axis_size = _get_size(src_dims, axis)
    dst = [0] * (outer_size * inner_size * axis_size)

    for ou in range(outer_size):
        for inn in range(inner_size):
            space_denom = 0.
            space_max = -3.402823466e+38

            ou_in_offset = ou * axis_size * inner_size + inn

            for ax in range(axis_size):
                idx = ou_in_offset + ax * inner_size
                space_max = max(space_max, src[idx])
            
            for ax in range(axis_size):
                idx = ou_in_offset + ax * inner_size
                dst[idx] = exp(src[idx] - space_max)
                space_denom += dst[idx]

            if space_denom:
                space_denom = 1. / space_denom
            else:
                space_denom = 1.
            
            for ax in range(axis_size):
                idx = ou_in_offset + ax * inner_size
                dst[idx] *= space_denom
    return dst

def _product(dims: dims_t):
    res = 1
    for dim in dims:
        res *= dim
    return res

def test():
    dims = [2,2,2,2]
    src = [i for i in range(_product(dims))]
    axis = 1
    print(softmax(src, dims, axis))
    try:
        import torch
        src_torch = torch.tensor(src,dtype=torch.float).reshape(dims)
        print(torch.nn.Softmax(dim=1)(src_torch))
    except:
        print('not install torch')
if __name__ == '__main__':
    test()