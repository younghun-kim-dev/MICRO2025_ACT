# QKV ISA Exercise 1

## What I did
- Defined the QKV accelerator ISA with the TAIDL Python API.
- Ran `python QKV.py` to generate the oracle (`targets/QKV/oracle/`) and confirm the Python API.

## Steps
- From `tutorials/micro25/` on the host:
  - `./copy.sh exercise1` → created `QKV.py` at the root of the `act/` repo.
- Opened `QKV.py` and confirmed:
  - `d1`: 128×64, bf16 (main I/O scratchpad)
  - `d2`: 64×64, bf16 (intermediate/compute buffer)
  - implicit `d0` (off-chip) used for load/store
  - 7 instructions:
    - `load_rm`
    - `load_cm` (load + transpose)
    - `store_rm`
    - `store_cm` (store + transpose)
    - `mov`
    - `gemm` (fixed 64×64)
    - `softmax` (row-wise on n×64)
- Started the container:
  - `cd tutorials/micro25/`
  - `./docker.sh --sim`
- Inside `/workspace`:
  - `python QKV.py`

## Result
https://github.com/user-attachments/assets/67203785-be4c-4222-9dd8-338cbb330d68


- Generated: `/workspace/targets/QKV/oracle/`
  - `api.py`, `decorator.py`, `utils.py`, `build/`
- Console: `Oracle build complete for QKV`

## Key points
- Load/store already cover **row-major vs column-major** via transpose, so kernels don’t need an extra transpose op.
- GEMM is **tile-based** (fixed 64×64) and writes to `d2`.
- Softmax is **parametric** (`n×64`), so it can run on partial rows, not only full 64×64 blocks.
