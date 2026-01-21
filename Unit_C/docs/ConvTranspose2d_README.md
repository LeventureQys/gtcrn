# ConvTranspose2d C Implementation

Detailed C implementation of PyTorch's `nn.ConvTranspose2d` used in the GTCRN decoder.

## Overview

ConvTranspose2d (also called Transposed Convolution or Deconvolution) is used for **upsampling** feature maps. It's the opposite operation of regular convolution.

### Key Differences from Conv2d

| Aspect | Conv2d | ConvTranspose2d |
|--------|--------|-----------------|
| Purpose | Downsampling | Upsampling |
| Output Size | Smaller or same | Larger or same |
| Formula | `(input - kernel + 2*pad) / stride + 1` | `(input - 1) * stride - 2*pad + kernel` |
| Operation | Many-to-one | One-to-many |

## Usage in GTCRN

From [gtcrn1.py](../gtcrn1.py), ConvTranspose2d is used in the decoder:

### Line 251-255: Decoder Layers
```python
self.de_convs = nn.ModuleList([
    GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*5,1), dilation=(5,1), use_deconv=True),
    GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True),
    GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*1,1), dilation=(1,1), use_deconv=True),
    ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
    ConvBlock(16, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
])
```

### Purpose in GTCRN
1. **Restore frequency resolution**: Encoder downsamples from 385 bins → 97 bins
2. **Decoder upsamples back**: 97 bins → 194 bins → 385 bins
3. **Final output**: 2 channels (real + imaginary mask)

## C Implementation

### Function Signature

```c
void conv2d_transpose_forward(
    const Tensor* input,
    Tensor* output,
    const Conv2dParams* params
);
```

### Algorithm

```c
// For each input position (ih, iw):
//   For each kernel position (kh, kw):
//     Calculate output position:
//       oh = ih * stride_h - padding_h + kh * dilation_h
//       ow = iw * stride_w - padding_w + kw * dilation_w
//
//     If output position is valid:
//       output[oh, ow] += input[ih, iw] * weight[kh, kw]
```

### Key Implementation Details

1. **Initialize output to zero** (accumulation operation)
2. **Iterate over input pixels** (not output pixels like Conv2d)
3. **Each input spreads to multiple outputs** based on kernel size
4. **Stride controls spacing** between output positions

## Examples

### Example 1: Basic 2x Upsampling

```c
// Input: [1, 1, 4, 4]
// Kernel: 4x4, Stride: 2, Padding: 1
// Output: [1, 1, 8, 8]

Tensor* input = tensor_create(1, 1, 4, 4);
int out_h = calculate_transpose_output_size(4, 4, 2, 1, 1);  // = 8
int out_w = calculate_transpose_output_size(4, 4, 2, 1, 1);  // = 8
Tensor* output = tensor_create(1, 1, out_h, out_w);

Conv2dParams params;
params.kernel_h = 4;
params.kernel_w = 4;
params.stride_h = 2;
params.stride_w = 2;
params.padding_h = 1;
params.padding_w = 1;
params.dilation_h = 1;
params.dilation_w = 1;
params.groups = 1;
params.in_channels = 1;
params.out_channels = 1;
params.use_bias = 0;

// Allocate and initialize weights
params.weight = (float*)malloc(1 * 1 * 4 * 4 * sizeof(float));

conv2d_transpose_forward(input, output, &params);
```

### Example 2: GTCRN Decoder Block (Line 254)

```c
// ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True)
// Input: [1, 16, 63, 97]
// Output: [1, 16, 63, 194]

Tensor* input = tensor_create(1, 16, 63, 97);

int out_h = calculate_transpose_output_size(63, 1, 1, 0, 1);  // = 63
int out_w = calculate_transpose_output_size(97, 5, 2, 2, 1);  // = 194

Tensor* output = tensor_create(1, 16, out_h, out_w);

Conv2dParams params;
params.kernel_h = 1;
params.kernel_w = 5;
params.stride_h = 1;
params.stride_w = 2;
params.padding_h = 0;
params.padding_w = 2;
params.dilation_h = 1;
params.dilation_w = 1;
params.groups = 2;  // Grouped convolution
params.in_channels = 16;
params.out_channels = 16;
params.use_bias = 1;

// Weight size: 16 * (16/2) * 1 * 5 = 640
int weight_size = 16 * 8 * 1 * 5;
params.weight = (float*)malloc(weight_size * sizeof(float));
params.bias = (float*)malloc(16 * sizeof(float));

conv2d_transpose_forward(input, output, &params);

// Apply BatchNorm + PReLU
batch_norm_2d_forward(output, gamma, beta, mean, var, 1e-5f);
prelu_forward(output, prelu_weights);
```

### Example 3: Final Decoder Layer (Line 255)

```c
// ConvBlock(16, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
// Input: [1, 16, 63, 194]
// Output: [1, 2, 63, 385]  <- Complex mask (real + imag)

Tensor* input = tensor_create(1, 16, 63, 194);

int out_h = calculate_transpose_output_size(63, 1, 1, 0, 1);  // = 63
int out_w = calculate_transpose_output_size(194, 5, 2, 2, 1); // = 385

Tensor* output = tensor_create(1, 2, out_h, out_w);

Conv2dParams params;
params.kernel_h = 1;
params.kernel_w = 5;
params.stride_h = 1;
params.stride_w = 2;
params.padding_h = 0;
params.padding_w = 2;
params.dilation_h = 1;
params.dilation_w = 1;
params.groups = 1;
params.in_channels = 16;
params.out_channels = 2;  // Real and imaginary
params.use_bias = 1;

// Weight size: 2 * 16 * 1 * 5 = 160
int weight_size = 2 * 16 * 1 * 5;
params.weight = (float*)malloc(weight_size * sizeof(float));
params.bias = (float*)malloc(2 * sizeof(float));

conv2d_transpose_forward(input, output, &params);

// Apply BatchNorm + Tanh (is_last=True)
batch_norm_2d_forward(output, gamma, beta, mean, var, 1e-5f);
tanh_forward(output);  // Output range: [-1, 1]
```

## Output Size Calculation

### Formula
```
output_size = (input_size - 1) * stride - 2 * padding + kernel_size
```

### C Implementation
```c
int calculate_transpose_output_size(
    int input_size,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int effective_kernel = dilation * (kernel_size - 1) + 1;
    return (input_size - 1) * stride - 2 * padding + effective_kernel;
}
```

### Examples

| Input | Kernel | Stride | Padding | Output | Upsampling Factor |
|-------|--------|--------|---------|--------|-------------------|
| 4 | 4 | 2 | 1 | 8 | 2x |
| 8 | 4 | 2 | 1 | 16 | 2x |
| 97 | 5 | 2 | 2 | 194 | ~2x |
| 194 | 5 | 2 | 2 | 385 | ~2x |

## Grouped ConvTranspose2d

When `groups > 1`, channels are divided into independent groups:

```c
int in_channels_per_group = in_channels / groups;
int out_channels_per_group = out_channels / groups;

// Each group processes independently
for (int g = 0; g < groups; g++) {
    // Process channels [g*cpg : (g+1)*cpg]
}
```

### GTCRN Usage
- Line 254: `groups=2` with 16 channels
  - Group 0: channels 0-7
  - Group 1: channels 8-15
  - Reduces parameters: 16×16×5 → 16×8×5

## Visualization

### Regular Conv2d (Downsampling)
```
Input (4x4)          Output (2x2)
┌─┬─┬─┬─┐           ┌───┬───┐
│1│2│3│4│           │ A │ B │
├─┼─┼─┼─┤    →      ├───┼───┤
│5│6│7│8│           │ C │ D │
├─┼─┼─┼─┤           └───┴───┘
│9│0│1│2│
├─┼─┼─┼─┤
│3│4│5│6│
└─┴─┴─┴─┘
```

### ConvTranspose2d (Upsampling)
```
Input (2x2)          Output (4x4)
┌───┬───┐           ┌─┬─┬─┬─┐
│ A │ B │           │*│*│*│*│
├───┼───┤    →      ├─┼─┼─┼─┤
│ C │ D │           │*│*│*│*│
└───┴───┘           ├─┼─┼─┼─┤
                    │*│*│*│*│
                    ├─┼─┼─┼─┤
                    │*│*│*│*│
                    └─┴─┴─┴─┘
```

## Building and Running

### Compile
```bash
cd Unit_C
make -f Makefile_transpose
```

### Run Examples
```bash
./conv_transpose2d_example
```

### Expected Output
```
Example 1: Basic ConvTranspose2d - 2x Upsampling
Example 2: GTCRN Decoder ConvTranspose2d Block
Example 3: GTCRN Final Decoder Layer
Example 4: Understanding Stride in ConvTranspose2d
```

## Performance Considerations

### Memory Access Pattern
- ConvTranspose2d has **scattered writes** (vs. Conv2d's sequential writes)
- Each input pixel writes to multiple output locations
- Less cache-friendly than regular convolution

### Optimization Strategies
1. **Reorder loops** for better cache locality
2. **Batch processing** to amortize overhead
3. **SIMD vectorization** for inner loops
4. **Pre-compute output indices** to reduce calculations

### Complexity
- Time: O(B × C_in × C_out × H_in × W_in × K_h × K_w)
- Space: O(B × C_out × H_out × W_out)

## Comparison with PyTorch

### PyTorch Code
```python
import torch.nn as nn

# GTCRN decoder layer
deconv = nn.ConvTranspose2d(
    in_channels=16,
    out_channels=16,
    kernel_size=(1, 5),
    stride=(1, 2),
    padding=(0, 2),
    groups=2
)

input = torch.randn(1, 16, 63, 97)
output = deconv(input)  # [1, 16, 63, 194]
```

### C Code
```c
Tensor* input = tensor_create(1, 16, 63, 97);
Tensor* output = tensor_create(1, 16, 63, 194);

Conv2dParams params = {
    .kernel_h = 1, .kernel_w = 5,
    .stride_h = 1, .stride_w = 2,
    .padding_h = 0, .padding_w = 2,
    .groups = 2,
    .in_channels = 16, .out_channels = 16,
    .weight = weights, .bias = bias,
    .use_bias = 1
};

conv2d_transpose_forward(input, output, &params);
```

## Common Pitfalls

1. **Forgetting to zero output**: ConvTranspose2d accumulates, must initialize to 0
2. **Wrong weight indexing**: Weight layout differs from Conv2d
3. **Incorrect output size**: Use `calculate_transpose_output_size()`
4. **Groups parameter**: Must divide both in_channels and out_channels evenly

## Testing

The example file includes 4 comprehensive tests:

1. **Basic upsampling** - Simple 2x upsampling
2. **GTCRN decoder block** - Real decoder configuration
3. **Final layer** - Complex mask generation
4. **Stride comparison** - Understanding stride effects

Run with:
```bash
make -f Makefile_transpose run
```

## References

- GTCRN Paper: "GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources"
- PyTorch Documentation: [nn.ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
- Source: [gtcrn1.py](../gtcrn1.py) lines 96-104, 247-262

## Related Files

- [conv2d.h](conv2d.h) - Header with declarations
- [conv2d.c](conv2d.c) - Implementation (lines 113-173)
- [conv_transpose2d_example.c](conv_transpose2d_example.c) - Examples and tests
- [test_conv2d.c](test_conv2d.c) - General convolution tests
