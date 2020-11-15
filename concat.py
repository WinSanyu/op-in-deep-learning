'''
Refer to https://oneapi-src.github.io/oneDNN/
'''
from typing import List, Tuple
dims_t = List[int]
tensor_t = List[float]

def _get_size(src_dims: List[dims_t], axis: int) -> Tuple[int, int, int]:
    # TODO: assert sdims

    outer_size = 1
    inner_size = 1
    axis_size = 0
    
    for i in range(axis):
        outer_size *= src_dims[0][i]
    for i in range(axis+1, len(src_dims[0])):
        inner_size *= src_dims[0][i]
    for i in range(len(src_dims)):
        axis_size += src_dims[i][axis]
    
    return outer_size, inner_size, axis_size

def concat(srcs: List[tensor_t], src_dims: List[dims_t], axis: int) -> tensor_t:
    outer_size, inner_size, axis_size = _get_size(src_dims, axis)
    dst = [0] * (outer_size * inner_size * axis_size)
    for ou in range(outer_size):
        for inn in range(inner_size):
            off_dst = ou * axis_size * inner_size
            for i_input in range(len(srcs)):
                src = srcs[i_input]
                i_axis_size = src_dims[i_input][axis]
                off_src = ou * i_axis_size * inner_size

                for axis_ind in range(i_axis_size):
                    idx = axis_ind * inner_size + inn
                    dst[off_dst + idx] = src[off_src + idx]
                
                off_dst += i_axis_size * inner_size
    
    return dst

def _product(dims: dims_t):
    res = 1
    for dim in dims:
        res *= dim
    return res

def test():
    dims = [[2,2,2,1], [2,1,2,1]]
    srcs = [[i for i in range(_product(dims[0]))], [-i for i in range(_product(dims[1]))]]
    axis = 1
    print(concat(srcs, dims, axis))

if __name__ == '__main__':
    test()