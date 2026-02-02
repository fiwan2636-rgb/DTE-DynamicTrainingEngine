# Adaptive Training Engine

This repository provides a **generic, block-based engine for training neural networks
with adaptive and recursive execution**. It is designed as a reusable control layer
for iteration, stopping, and unrolling during training, without hard-coding a
specific algorithm.

The engine includes a **Tiny Recursive Model (TRM)** implementation as a **baseline
reference**

## Quick launch

python pretrain.py

## Quick launch

```bash
python pretrain.py
```
This command launches training of the Tiny Recursive Model (TRM) on a vector
sorting task.

## Reference implementation

The TRM baseline follows the original implementation and design described in:

https://github.com/SamsungSAILMontreal/TinyRecursiveModels

The provided TRM baseline reproduces the reported performance on Sudoku (~4h00 on a 
single A100)

## Acknowledgements

Special thanks to **Alexia Jolicoeur-Martineau** for her help and guidance in
implementing this system properly and ensuring reproducibility of the original TRM
performance.
