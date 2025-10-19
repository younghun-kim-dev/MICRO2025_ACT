# QKV ISA Exercise 3 – Backend Generation Notes

## Goal
- Take the QKV ISA from Exercise 1 and make ACT generate a **real compiler backend**.
- Write attention once in **HLO**.
- Let the generated backend lower HLO → QKV assembly.
- Run the same `test_qkv.py` as in Exercise 2 to check against **FPGA golden**.

## What I added
- Opened `QKV.py` (from Ex1) and added at the end:

  - `qkv.generate_oracle()`   ← already there from Ex1  
  - `qkv.generate_backend()`  ← **added for Ex3**

So now the ISA file generates **both** oracle and backend.

## Commands I ran
- Host:
  - `cd tutorials/micro25/`
  - `./copy.sh exercise3`   (this gave me `attention.hlo` and test script)
  - `./docker.sh --compile`
- Container (`/workspace`):
  - `python QKV.py`   ← this is the step that actually builds the backend

## What got generated
- Directory: `/workspace/targets/QKV/backend/`
  - Rust side: e-graph / isel rewrites
  - C++ side: allocator (`instructions.h`, `globals.cc`, etc.)
- Binary: `/workspace/backends/QKV`
- Console said things like:
  - `Copied generic backend structure ...`
  - `Backend generation complete for QKV`
  - `Backend build complete for QKV`

(So: same pattern as oracle, but now also “backend”.)

https://github.com/user-attachments/assets/123eea8f-bfbf-4a79-ad3e-3f4aa989b0e4  

## HLO I wrote
- File: `attention.hlo`
- Shape: all `bf16[64,64]`
- Steps:
  1. `k_transpose = transpose(k), dimensions={1,0}`
  2. `scores = dot(q, k_transpose)`
  3. `scores_exp = exponential(scores)`
  4. `sum = reduce(scores_exp, 0), dimensions={1}`
  5. `sum_broadcast = broadcast(sum), dimensions={0}`
  6. `probs = divide(scores_exp, sum_broadcast)`
  7. `output = dot(probs, v)`  ← ROOT

(= the same math as Exercise 2, but now in HLO, not hand-written API calls.)

## Compiling HLO → QKV asm
- Still inside container:
  - `./backends/QKV --input attention.hlo --output asm/compiled_qkv.py`
- Output file looked like the manual kernel from Ex2:
  - load Q
  - load K (column-major)
  - gemm
  - softmax
  - mov
  - load V
  - gemm
  - mov
  - store

So the **compiler basically rediscovered** the 9-instruction pattern from Exercise 2.

https://github.com/user-attachments/assets/2f28aa09-3e96-4992-966d-b7ad73af40a0  

## Testing
https://github.com/user-attachments/assets/5e2cad9f-b368-4606-a848-bdddc23d548b

- Run container in sim mode:
  - `./docker.sh --sim`
  - `python test_qkv.py`
- Expected (same as Ex2):
  - `Output matches golden exactly!`
  - `Max absolute difference ... 0.0`

## Important facts
- Backend is generated **from the ISA file** (no hand-written compiler).
- Instruction selection = Rust e-graph; memory allocation = C++ solver.
- HLO only used standard ops: `transpose`, `dot`, `exponential`, `reduce`, `broadcast`, `divide`.
- Addresses (0, 8192, 16384, 24576) and HBM size (32768) are the same as Exercise 2, so the test script works unchanged.
