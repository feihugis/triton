import numpy as np
import torch
import triton
import triton.language as tl
import pytest

import torch

import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1},
                                      num_stages=num_stages, num_warps=num_warps))
                    # split_k
                    for split_k in [2, 4, 8, 16]:
                        configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k},
                                                     num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        # good for int8
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
    ] + get_configs_io_bound(),
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10
    },
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def _kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr,
            ACC_TYPE: tl.constexpr
            ):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K * SPLIT_K):
        # if EVEN_K:
        #     a = tl.load(A)
        #     b = tl.load(B)
        # else:
        #     a = tl.load(A, mask=rk[None, :] < k, other=0.)
        #     b = tl.load(B, mask=rk[:, None] < k, other=0.)

        if EVEN_K:
            a_int8 = tl.load(A)
            a = a_int8.to(tl.float8, bitcast=True).to(tl.float16)
            b_int8 = tl.load(B)
            b = b_int8.to(tl.float8, bitcast=True).to(tl.float16)
        else:
            a_int8 = tl.load(A, mask=rk[None, :] < k, other=0.)
            a = a_int8.to(tl.float8, bitcast=True).to(tl.float16)
            b_int8 = tl.load(B, mask=rk[:, None] < k, other=0.)
            b = b_int8.to(tl.float8, bitcast=True).to(tl.float16)

        # tl.printf("a++++++\n\n++++++a\n", a)
        # tl.printf("b++++++\n\n++++++b\n", b)
        acc += tl.dot(a, b)
        # tl.debug_barrier()
        # tl.printf("++++++\n\n++++++\n", acc)
        # tl.printf("acc = %f", acc)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(tl.float8).to(C.dtype.element_ty, bitcast=True)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


class _matmul(torch.autograd.Function):
    kernel = _kernel

    _locks = dict()

    @staticmethod
    def _call(a, b):
        device = a.device
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape
        # allocates output
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        # accumulator types
        ACC_TYPE = tl.float32 #if a.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32
        # launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
        _kernel[grid](a, b, c, M, N, K,
                      a.stride(0), a.stride(1),
                      b.stride(0), b.stride(1),
                      c.stride(0), c.stride(1),
                      GROUP_M=8, ACC_TYPE=ACC_TYPE)
        return c

    @staticmethod
    def forward(ctx, a, b):
        return _matmul._call(a, b)


matmul = _matmul.apply


@triton.jit
def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # tl.printf("BLOCKSIZE: ", BLOCK_SIZE)
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = input
    tl.store(output_ptr + offsets, output, mask=mask)
    # o = tl.load(output_ptr + offsets, mask=mask).to(tl.float8)
    # tl.printf("++++++", o)

def fp16_2_fp8(f16_tensor):
    n_elements = f16_tensor.numel()
    f8_output_tensor = torch.empty_like(f16_tensor, dtype=torch.int8)
    f8_output = triton.reinterpret(f8_output_tensor, tl.float8)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    copy_kernel[grid](f16_tensor, f8_output, n_elements, BLOCK_SIZE=1024)
    # print("f8_output = ", f8_output)
    return f8_output_tensor

def fp8_2_fp16(f8_tensor):
    n_elements = f8_tensor.numel()
    f8_triton = triton.reinterpret(f8_tensor, tl.float8)

    f16_output_tensor = torch.empty_like(f8_tensor, dtype=torch.float16)
    f16_output = triton.reinterpret(f16_output_tensor, tl.float16)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    copy_kernel[grid](f8_triton, f16_output, n_elements, BLOCK_SIZE=1024)
    return f16_output_tensor

def test_matmul():
    torch.manual_seed(0)
    M = 768
    N = 768
    K = 768*4
    AT = False
    BT = True
    DTYPE=torch.float16
    a = 0.01 * torch.ones((K, M) if AT else (M, K), device="cuda", dtype=DTYPE)
    b = 0.05 * torch.ones((N, K) if BT else (K, N), device="cuda", dtype=DTYPE)
    a = a.t() if AT else a
    b = b.t() if BT else b

    th_c = torch.matmul(a, b)
    # tt_c = triton.testing.catch_oor(lambda: triton.ops.matmul(a, b), pytest)
    # triton.testing.assert_almost_equal(th_c, tt_c)

    print(th_c)
    # print(tt_c)

      # create fp8 tensor
    a_fp8 = fp16_2_fp8(a)
    b_fp8 = fp16_2_fp8(b)
    # a_fp8 = 120 * torch.ones((K, M) if AT else (M, K), device="cuda", dtype=torch.int8) #fp16_2_fp8(a)
    # b_fp8 = 120 * torch.ones((K, M) if AT else (M, K), device="cuda", dtype=torch.int8) #fp16_2_fp8(b)
    # print(a_fp8)
    # b_fp16 = fp8_2_fp16(b_fp8)
    # print(a_fp8)
    c_fp8 = matmul(a_fp8, b_fp8)
    print(c_fp8)
    c_fp16 = fp8_2_fp16(c_fp8)
    print(c_fp16)
    # print(b_fp16)
    # print(b)
    # c_fp8_tensor = torch.empty_like(tt_c, dtype=torch.int8)
    # c_fp8 = triton.reinterpret(c_fp8_tensor, tl.float8)

    # @triton.jit
    # def matmul_fp8(a_fp8, b_fp8, c_fp8):
    #   a = tl.load(a_fp8).to(tl.float16)
    #   b = tl.load(b_fp8).to(tl.float16)
    #   c = triton.ops.matmul(a, b)
    #   tl.store(c, c_fp8)

    # matmul_fp8[(1, )](a_fp8, b_fp8, c_fp8)
    # c_fp8 = triton.ops.matmul(a_fp8, b_fp8)
    # c_fp16 = fp8_2_fp16(c_fp8)

    # print(c_fp16)




def test_f16_to_f8_rounding():
    """Takes all float16s, converts them to float8 and back to float16. Checks that the absolute
    error is the minimum over all float8.
    Or the same explanation a bit mathier:
    for all f16 |f16 - fromf8(tof8(f16))| == min over all f8 |f16 - fromf8(f8)|"""
    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    # torch.view with a dtype isn't supported in triton's torch yet so use numpy's view
    f16_input_np = (
        np.array(
            range(-int(2 ** (16 - 1)), int(2 ** (16 - 1))), dtype=np.int16,
        )
        .view(np.float16)
    )
    f16_input = torch.tensor(f16_input_np, dtype=torch.float16, device='cuda')
    n_elements = f16_input.numel()
    f8_output_tensor = torch.empty_like(f16_input, dtype=torch.int8)
    f8_output = triton.reinterpret(f8_output_tensor, tl.float8)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    copy_kernel[grid](f16_input, f8_output, n_elements, BLOCK_SIZE=1024)

    f16_output = torch.empty_like(f16_input, dtype=torch.float16)
    copy_kernel[grid](f8_output, f16_output, n_elements, BLOCK_SIZE=1024)

    abs_error = torch.abs(f16_input - f16_output)

    all_f8_vals_tensor = torch.tensor(range(2 ** 8), dtype=torch.uint8, device='cuda')
    all_f8_vals = triton.reinterpret(all_f8_vals_tensor, tl.float8)
    all_f8_vals_in_f16 = torch.empty_like(all_f8_vals_tensor, dtype=torch.float16)
    copy_kernel[grid](all_f8_vals, all_f8_vals_in_f16, n_elements=256, BLOCK_SIZE=1024)

    all_finite_f8_vals_in_f16 = all_f8_vals_in_f16[
        torch.isfinite(all_f8_vals_in_f16)
    ]

    min_error = torch.min(
        torch.abs(
            f16_input.reshape((-1, 1))
            - all_finite_f8_vals_in_f16.reshape((1, -1))
        ),
        dim=1,
    )[0]
    # 1.9375 is float8 max
    mismatch = torch.logical_and(
        abs_error != min_error, torch.logical_and(torch.isfinite(f16_input), torch.abs(f16_input) < 1.9375)
    )
    print("++++++")
    assert torch.all(
        torch.logical_not(mismatch)
    ), f"f16_input[mismatch]={f16_input[mismatch]} f16_output[mismatch]={f16_output[mismatch]} abs_error[mismatch]={abs_error[mismatch]} min_error[mismatch]={min_error[mismatch]}"

# test_f16_to_f8_rounding()
test_matmul()