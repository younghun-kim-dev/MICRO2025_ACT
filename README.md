# MICRO'25 ACT Tutorial – QKV Exercises (1–3)

## Exercise 1 — QKV ISA + Oracle
- Defined the QKV accelerator ISA using the TAIDL Python API.
- Confirmed the ISA scratchpad layout:
  - `d1`: 128×64, bf16 (main I/O scratchpad)
  - `d2`: 64×64, bf16 (intermediate/compute buffer)
- Generated the oracle (`targets/QKV/oracle/`) by running `python QKV.py`.

**Video:** [Exercise1](./video/Exercise1.mkv)

---

## Exercise 2 — Kernel Programming: Attention
- Used the generated Python APIs and the `@kernel(...)` decorator to write accelerator kernels.
- Implemented attention:
  - `Attention(Q, K, V) = softmax(Q × Kᵀ) × V`
- Verified the output against FPGA golden data with `python test_qkv.py`.

**Video:** [Exercise2](./video/Exercise2.mkv)

---

## Exercise 3 — Backend Generation + HLO → QKV Assembly
- Extended the ISA file (`QKV.py`) to generate both:
  - `qkv.generate_oracle()`
  - `qkv.generate_backend()`
- Built the backend and compiled an HLO implementation of attention (`attention.hlo`) down to QKV assembly.
- Re-verified with the same `python test_qkv.py` against FPGA golden data.

**Videos:**
- [Exercise3-1](./video/Exercise3-1.mkv)
- [Exercise3-2](./video/Exercise3-2.mkv)
- [Exercise3-3](./video/Exercise3-3.mkv)
