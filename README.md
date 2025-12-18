# MICRO'25 ACT Tutorial – QKV Exercises (1–3)

## Exercise 1 — QKV ISA + Oracle
- Defined the QKV accelerator ISA using the TAIDL Python API.
- Confirmed the ISA scratchpad layout:
  - `d1`: 128×64, bf16 (main I/O scratchpad)
  - `d2`: 64×64, bf16 (intermediate/compute buffer)
- Generated the oracle (`targets/QKV/oracle/`) by running `python QKV.py`.

https://github.com/user-attachments/assets/67203785-be4c-4222-9dd8-338cbb330d68

---

## Exercise 2 — Kernel Programming: Attention
- Used the generated Python APIs and the `@kernel(...)` decorator to write accelerator kernels.
- Implemented attention:
  - `Attention(Q, K, V) = softmax(Q × Kᵀ) × V`
- Verified the output against FPGA golden data with `python test_qkv.py`.

https://github.com/user-attachments/assets/f7db9599-c22a-49d6-a04e-4fcdccf767bb

---

## Exercise 3 — Backend Generation + HLO → QKV Assembly
- Extended the ISA file (`QKV.py`) to generate both:
  - `qkv.generate_oracle()`
  - `qkv.generate_backend()`
- Built the backend and compiled an HLO implementation of attention (`attention.hlo`) down to QKV assembly.
- Re-verified with the same `python test_qkv.py` against FPGA golden data.

https://github.com/user-attachments/assets/123eea8f-bfbf-4a79-ad3e-3f4aa989b0e4
https://github.com/user-attachments/assets/2f28aa09-3e96-4992-966d-b7ad73af40a0
https://github.com/user-attachments/assets/5e2cad9f-b368-4606-a848-bdddc23d548b
