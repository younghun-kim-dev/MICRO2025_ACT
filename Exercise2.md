# QKV ISA Exercise 2 – Kernel Programming Notes

## Goal
- Use the QKV ISA + Python APIs generated in Exercise 1.
- Write accelerator kernels with the `@kernel(...)` decorator.
- Implement full attention:
  - `Attention(Q, K, V) = softmax(Q × Kᵀ) × V`
- Verify against FPGA golden data with `python test_qkv.py`.

## Files created by `./copy.sh exercise2`
- `asm/identity.py` (done example)
- `asm/matmul.py` (done example)
- `asm/softmax.py` (done example)
- `asm/attention.py` (**to implement**)
- `data/Q.dat`, `data/K.dat`, `data/V.dat`, `data/attention.dat`
- `test_qkv.py`

So: 3 reference kernels, 1 incomplete kernel, test script, and real FPGA data.

## HBM layout (as used in the exercise)
- `0` → Q (64×64, bf16, 8192 B)
- `8192` → K
- `16384` → V
- `24576` → output
- total HBM in decorator: `32768` bytes

## Scratchpad plan
- `d1[0:63]` → first operand (Q, later P)
- `d1[64:127]` → second operand (Kᵀ first, later V)
- `d2[0:63]` → GEMM / softmax result

## What the 3 example kernels show
- **identity.py**: load 64 rows → store 64 rows (HBM ↔ d1 only)
- **matmul.py**: load A to `d1[0:63]`, load B to `d1[64:127]`, `gemm(...)` → `d2`, then `mov` to `d1`, then store
- **softmax.py**: because softmax runs on `d2`, they first do a GEMM to move data into `d2`, then softmax, then `mov` + store

Key constraint repeated in all three:
- GEMM output is always in **d2**
- softmax runs only on **d2**
- store reads from **d1**
→ therefore: “compute in d2 → mov to d1 → store” is the pattern

## Final attention kernel logic
1. Load Q (row-major) to `d1[0:63]`
2. Load K (column-major) to `d1[64:127]` → this gives **Kᵀ**
3. `gemm(addr_1=0, addr_2=64, addr_out=0)`  
   → `d2[0:63] = Q × Kᵀ`
4. `api.softmax(n=64, addr=0)`  
   → in-place on `d2[0:63]`, now this is **P**
5. `api.mov(n=64, addr_in=0, addr_out=0)`  
   → P goes to `d1[0:63]`
6. Load V (row-major) to `d1[64:127]` (reuse the Kᵀ space)
7. `api.gemm(addr_1=0, addr_2=64, addr_out=0)`  
   → `d2[0:63] = P × V` (final O)
8. `api.mov(n=64, addr_in=0, addr_out=0)`  
   → move O to `d1[0:63]`
9. `api.store_rm(n=64, addr_in=0, addr_out=24576)`  
   → write output to HBM

(That’s the 9-instruction sequence mentioned in the exercise.)

## Testing
https://github.com/user-attachments/assets/f7db9599-c22a-49d6-a04e-4fcdccf767bb


- On host:
  - `cd tutorials/micro25/`
  - `./docker.sh --sim`
- In container (`/workspace`):
  - `python test_qkv.py`
- Success message (example):
  - `Output matches golden exactly!`
  - `Max absolute difference ... 0.0`

## Important facts
- `load_cm` must be used for **K** to get **Kᵀ** directly.
- GEMM always writes to `d2`, never to `d1`.
- Softmax is in-place on `d2`.
- Store instructions read from `d1`, so results from `d2` must be moved first.
- `d1[64:127]` can be reused after the first GEMM (Kᵀ → V).
- FPGA `.dat` files in `data/` are the hardware reference outputs.
