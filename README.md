# MIT 6.S184: Generative AI with Stochastic Differential Equations

This repository contains the lab assignments for MIT 6.S184: Generative AI with Stochastic Differential Equations.

## Repository Structure

This repository is organized into two main components:

### 1. Labs with Unit Tests
- `lab_one.ipynb` - Lab 1: Simulating ODEs and SDEs
- `lab_two.ipynb` - Lab 2: (Description TBD)
- `lab_three.ipynb` - Lab 3: A Conditional Generative Model for Images

Each lab notebook includes comprehensive unit tests that I've added to help verify the correctness of implementations. These tests check:
- Mathematical formula correctness
- Shape consistency across different batch sizes
- Edge cases and boundary conditions
- Statistical properties and convergence
- Integration with other components

### 2. Solutions
My personal solutions to the lab assignments are located in the `solutions/` directory.

## About the Unit Tests

The unit tests have been added to provide immediate feedback on implementation correctness. Each test cell is placed directly below the corresponding function that needs to be implemented, allowing you to:
- Verify your work immediately after completing each section
- Understand the expected behavior through comprehensive test cases
- Catch common mistakes and edge cases early

## Known Issues and Fixes

### Lab 2: Simulator.simulate() Bug (Fixed)

**Issue**: The original `Simulator.simulate()` method in `lab_two.ipynb` had a bug where it used `len(ts)` instead of `ts.shape[1]` to determine the number of integration steps.

```python
# Original buggy code (line ~336):
for t_idx in range(len(ts) - 1):  # Bug: len(ts) returns batch_size, not num_steps
```

Since `ts` has shape `(batch_size, num_timesteps, 1)`, `len(ts)` returns `batch_size` instead of `num_timesteps`, causing the integration to only perform `batch_size - 1` steps instead of the intended `num_timesteps - 1` steps.

**Fix**: Changed to use `ts.shape[1]` to correctly get the number of timesteps:

```python
# Fixed code:
for t_idx in range(ts.shape[1] - 1):
```

This bug caused the Unit Tests for Problem 2.3 to fail with large integration errors (~7.75 distance from target), even though the `conditional_vector_field` implementation was correct. The fix has been applied to `solutions/lab_two.ipynb`.

## Course Information

MIT 6.S184 focuses on generative AI techniques using stochastic differential equations, covering topics such as:
- Numerical methods for ODEs and SDEs
- Flow matching and diffusion models
- Conditional generation and classifier-free guidance
- Modern architectures like U-Net for image generation

---

**Note**: The original lab materials are from MIT 6.S184. The unit tests and solutions are my personal additions for learning purposes.