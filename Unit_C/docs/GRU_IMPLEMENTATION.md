# GRU Layer Implementation for GTCRN

## Overview

This document describes the complete GRU (Gated Recurrent Unit) implementation for the GTCRN speech enhancement model. The implementation is based on the PyTorch model in `gtcrn1.py` and provides a C implementation optimized for embedded systems.

## Architecture

### GTCRN Model Structure

```
Input: (B, 769, T, 2) - Complex spectrogram (48kHz)
  ↓
ERB Compression: 769 bins → 385 bins
  ↓
SFE: (B, 3, T, 385) → (B, 9, T, 385)
  ↓
Encoder: 5 layers → (B, 16, T, 97)
  ↓
DPGRNN Layer 1: (B, 16, T, 97) → (B, 16, T, 97)
  ↓
DPGRNN Layer 2: (B, 16, T, 97) → (B, 16, T, 97)
  ↓
Decoder: 5 layers → (B, 2, T, 385)
  ↓
ERB Decompression: 385 bins → 769 bins
  ↓
Complex Mask Application
  ↓
Output: (B, 769, T, 2) - Enhanced complex spectrogram
```

## GRU Implementation

### 1. Standard GRU Cell

The GRU cell implements the following equations:

```
z_t = σ(W_z * x_t + U_z * h_{t-1} + b_z)  // Update gate
r_t = σ(W_r * x_t + U_r * h_{t-1} + b_r)  // Reset gate
h̃_t = tanh(W_h * x_t + U_h * (r_t ⊙ h_{t-1}) + b_h)  // Candidate
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  // New hidden state
```

**Implementation:** `GRU.c:121-165` - `gru_cell_forward()`

**Parameters:**
- `input_size`: Dimension of input vector
- `hidden_size`: Dimension of hidden state
- Weight matrices: W_z, U_z, W_r, U_r, W_h, U_h
- Bias vectors: b_z, b_r, b_h

### 2. Grouped GRU (GRNN)

GRNN splits the input into 2 groups and processes them independently, reducing parameters by ~50%.

**From gtcrn1.py lines 156-184:**
```python
class GRNN(nn.Module):
    def __init__(self, input_size, hidden_size, ...):
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, ...)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, ...)
```

**Implementation:** `GRU.c:302-384` - `grnn_forward()`

**Process:**
1. Split input into 2 groups along feature dimension
2. Process each group with separate GRU
3. Concatenate outputs

### 3. Dual-Path Grouped RNN (DPGRNN)

DPGRNN processes the input in two paths:
- **Intra-RNN**: Bidirectional processing across frequency dimension
- **Inter-RNN**: Unidirectional processing across time dimension (causal)

**From gtcrn1.py lines 186-226:**
```python
class DPGRNN(nn.Module):
    def __init__(self, input_size, width, hidden_size):
        # Intra: Bidirectional GRNN
        self.intra_rnn = GRNN(input_size, hidden_size//2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size))

        # Inter: Unidirectional GRNN
        self.inter_rnn = GRNN(input_size, hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size))
```

**Implementation:** `gtcrn_model.c:323-466` - `dpgrnn_forward()`

## DPGRNN Forward Pass Details

### Input Shape: (B, C, T, F) where C=16, T=time, F=97

### Intra-RNN Path (Frequency Processing)

```
1. Permute: (B, C, T, F) → (B, T, F, C)
   - Rearrange to process frequency dimension

2. Reshape: (B, T, F, C) → (B*T, F, C)
   - Flatten batch and time for independent processing

3. Bidirectional GRNN: (B*T, F, C) → (B*T, F, C)
   - Process each (B*T) sample across frequency bins
   - Input size: 16, Hidden size: 8 (per direction)
   - Output size: 16 (8 forward + 8 backward)

4. Linear: (B*T, F, C) → (B*T, F, C)
   - Fully connected layer

5. Reshape: (B*T, F, C) → (B, T, F, C)
   - Restore batch and time dimensions

6. LayerNorm + Residual
   - Normalize over (F, C) dimensions
   - Add input for residual connection
```

### Inter-RNN Path (Time Processing)

```
1. Permute: (B, T, F, C) → (B, F, T, C)
   - Rearrange to process time dimension

2. Reshape: (B, F, T, C) → (B*F, T, C)
   - Flatten batch and frequency for independent processing

3. Unidirectional GRNN: (B*F, T, C) → (B*F, T, C)
   - Process each (B*F) sample across time steps
   - Input size: 16, Hidden size: 16
   - Causal (unidirectional) for real-time processing

4. Linear: (B*F, T, C) → (B*F, T, C)
   - Fully connected layer

5. Reshape: (B*F, T, C) → (B, F, T, C)
   - Restore batch and frequency dimensions

6. Permute: (B, F, T, C) → (B, T, F, C)
   - Restore original dimension order

7. LayerNorm + Residual
   - Normalize over (F, C) dimensions
   - Add previous output for residual connection
```

### Final Step

```
Permute: (B, T, F, C) → (B, C, T, F)
- Restore original tensor layout
```

## Parameter Counts

### DPGRNN Parameters (input_size=16, width=97, hidden_size=16)

**Intra-RNN (Bidirectional GRNN):**
- Group 1 (forward): input=8, hidden=4
  - W_z, W_r, W_h: 3 × (4 × 8) = 96
  - U_z, U_r, U_h: 3 × (4 × 4) = 48
  - b_z, b_r, b_h: 3 × 4 = 12
  - Total per group: 156 parameters
- Group 2 (forward): 156 parameters
- Group 1 (backward): 156 parameters
- Group 2 (backward): 156 parameters
- **Intra-RNN Total: 624 parameters**

**Inter-RNN (Unidirectional GRNN):**
- Group 1: input=8, hidden=8
  - W_z, W_r, W_h: 3 × (8 × 8) = 192
  - U_z, U_r, U_h: 3 × (8 × 8) = 192
  - b_z, b_r, b_h: 3 × 8 = 24
  - Total per group: 408 parameters
- Group 2: 408 parameters
- **Inter-RNN Total: 816 parameters**

**Linear Layers:**
- Intra FC: 16 × 16 = 256 parameters
- Inter FC: 16 × 16 = 256 parameters
- **Linear Total: 512 parameters**

**LayerNorm:**
- Intra LN: 97 × 16 × 2 = 3,104 parameters (weight + bias)
- Inter LN: 97 × 16 × 2 = 3,104 parameters
- **LayerNorm Total: 6,208 parameters**

**DPGRNN Total: 8,160 parameters per layer**
**Two DPGRNN layers: 16,320 parameters**

## Key Features

### 1. Grouped Processing
- Reduces parameters by 50% compared to standard GRU
- Maintains performance through independent group processing

### 2. Dual-Path Architecture
- **Intra-RNN**: Captures frequency correlations (bidirectional)
- **Inter-RNN**: Captures temporal dependencies (causal)

### 3. Residual Connections
- Helps gradient flow during training
- Preserves input information

### 4. Layer Normalization
- Stabilizes training
- Normalizes over (frequency, channel) dimensions

## Memory Requirements

For a single frame (B=1, T=63, F=97, C=16):

**Working Buffers:**
- x_btfc: 63 × 97 × 16 = 97,776 floats (391 KB)
- intra_out: 97,776 floats (391 KB)
- intra_x: 97,776 floats (391 KB)
- inter_in: 97,776 floats (391 KB)
- inter_out: 97,776 floats (391 KB)
- inter_x: 97,776 floats (391 KB)
- temp: 4 × 16 = 64 floats (256 bytes)

**Total Working Memory: ~2.3 MB per DPGRNN forward pass**

## Optimization Opportunities

### 1. In-Place Operations
- Reuse buffers where possible
- Reduce memory allocations

### 2. SIMD Vectorization
- Matrix-vector multiplications
- Element-wise operations

### 3. Fixed-Point Arithmetic
- Convert float32 to int16/int8
- Reduce memory bandwidth

### 4. Weight Quantization
- 8-bit or 4-bit quantization
- Minimal accuracy loss

## Usage Example

```c
// Create DPGRNN
DPGRNN* dpgrnn = dpgrnn_create(16, 97, 16);

// Load weights from file
// gru_weights_load(dpgrnn->intra_gru_g1_fwd, "intra_g1_fwd.bin");
// ... load other weights

// Prepare input tensor
Tensor input = {
    .data = input_data,
    .shape = {.batch = 1, .channels = 16, .height = 63, .width = 97}
};

Tensor output = {
    .data = output_data,
    .shape = {.batch = 1, .channels = 16, .height = 63, .width = 97}
};

// Forward pass
dpgrnn_forward(&input, &output, dpgrnn);

// Cleanup
dpgrnn_free(dpgrnn);
```

## Testing

Test files:
- `test_gru.c`: Unit tests for GRU cell and sequence processing
- `test_gtcrn_model.c`: Integration tests for full GTCRN model

## References

1. **GTCRN Paper**: "GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources"
2. **GRU Paper**: Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder"
3. **PyTorch Implementation**: `gtcrn1.py` (lines 156-226)

## File Structure

```
Unit_C/
├── GRU.h                   # GRU interface definitions
├── GRU.c                   # GRU implementation
├── gtcrn_model.h           # GTCRN model interface
├── gtcrn_model.c           # GTCRN model implementation (includes DPGRNN)
├── test_gru.c              # GRU unit tests
└── GRU_IMPLEMENTATION.md   # This documentation
```

## Notes

1. **Causality**: Inter-RNN is unidirectional (causal) to enable real-time processing
2. **Bidirectionality**: Intra-RNN is bidirectional as it processes frequency (not time)
3. **Weight Loading**: Weights must be loaded from trained PyTorch model
4. **Precision**: Current implementation uses float32; can be quantized for embedded systems
