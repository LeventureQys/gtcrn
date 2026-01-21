# GTCRN Conv2d C Implementation

This directory contains a C implementation of the Conv2d operations used in the GTCRN neural network model.

## Overview

The GTCRN model (from [gtcrn1.py](../gtcrn1.py)) uses several types of convolution operations:

1. **Regular Conv2d** - Standard 2D convolution
2. **Conv2d Transpose** - Deconvolution for upsampling
3. **Depthwise Conv2d** - Efficient convolution with groups=channels
4. **Pointwise Conv2d** - 1x1 convolution for channel mixing
5. **Batch Normalization** - Normalization layer
6. **Activation Functions** - PReLU and Tanh

## Files

- **conv2d.h** - Header file with function declarations and data structures
- **conv2d.c** - Implementation of all convolution operations
- **test_conv2d.c** - Test suite demonstrating usage
- **Makefile_conv2d** - Build configuration

## Data Structures

### Tensor
```c
typedef struct {
    float* data;
    TensorShape shape;
} Tensor;
```
Represents a 4D tensor with shape (batch, channels, height, width).

### Conv2dParams
```c
typedef struct {
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int padding_h, padding_w;
    int dilation_h, dilation_w;
    int groups;
    int in_channels, out_channels;
    float* weight;
    float* bias;
    int use_bias;
} Conv2dParams;
```
Contains all parameters needed for convolution operations.

## Key Functions

### Convolution Operations

#### conv2d_forward
```c
void conv2d_forward(
    const Tensor* input,
    Tensor* output,
    const Conv2dParams* params
);
```
Standard 2D convolution with support for:
- Arbitrary kernel sizes
- Stride and padding
- Dilation (atrous convolution)
- Grouped convolution
- Optional bias

#### conv2d_transpose_forward
```c
void conv2d_transpose_forward(
    const Tensor* input,
    Tensor* output,
    const Conv2dParams* params
);
```
Transposed convolution (deconvolution) for upsampling operations.

#### depthwise_conv2d_forward
```c
void depthwise_conv2d_forward(
    const Tensor* input,
    Tensor* output,
    const Conv2dParams* params
);
```
Efficient depthwise convolution where groups=in_channels.

#### pointwise_conv2d_forward
```c
void pointwise_conv2d_forward(
    const Tensor* input,
    Tensor* output,
    const Conv2dParams* params
);
```
1x1 convolution for channel-wise operations.

### Normalization and Activation

#### batch_norm_2d_forward
```c
void batch_norm_2d_forward(
    Tensor* input,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float eps
);
```
Batch normalization using pre-computed statistics.

#### prelu_forward
```c
void prelu_forward(
    Tensor* input,
    const float* weight
);
```
Parametric ReLU activation: f(x) = x if x > 0, else alpha * x

#### tanh_forward
```c
void tanh_forward(Tensor* input);
```
Hyperbolic tangent activation.

## Building and Running

### Compile
```bash
cd Unit_C
make -f Makefile_conv2d
```

### Run Tests
```bash
make -f Makefile_conv2d run
```

### Clean
```bash
make -f Makefile_conv2d clean
```

## Usage Example

```c
#include "conv2d.h"

// Create input tensor (batch=1, channels=3, height=32, width=32)
Tensor* input = tensor_create(1, 3, 32, 32);

// Fill with data
for (int i = 0; i < 1 * 3 * 32 * 32; i++) {
    input->data[i] = (float)rand() / RAND_MAX;
}

// Calculate output size
int out_h = calculate_output_size(32, 3, 1, 1, 1);
int out_w = calculate_output_size(32, 3, 1, 1, 1);

// Create output tensor
Tensor* output = tensor_create(1, 16, out_h, out_w);

// Setup convolution parameters
Conv2dParams params;
params.kernel_h = 3;
params.kernel_w = 3;
params.stride_h = 1;
params.stride_w = 1;
params.padding_h = 1;
params.padding_w = 1;
params.dilation_h = 1;
params.dilation_w = 1;
params.groups = 1;
params.in_channels = 3;
params.out_channels = 16;
params.use_bias = 1;

// Allocate and initialize weights
int weight_size = 16 * 3 * 3 * 3;
params.weight = (float*)malloc(weight_size * sizeof(float));
params.bias = (float*)malloc(16 * sizeof(float));

// Initialize weights (normally loaded from trained model)
for (int i = 0; i < weight_size; i++) {
    params.weight[i] = (float)rand() / RAND_MAX - 0.5f;
}

// Perform convolution
conv2d_forward(input, output, &params);

// Apply batch normalization
float gamma[16], beta[16], mean[16], var[16];
// ... initialize normalization parameters ...
batch_norm_2d_forward(output, gamma, beta, mean, var, 1e-5f);

// Apply activation
float prelu_weights[16];
// ... initialize prelu weights ...
prelu_forward(output, prelu_weights);

// Cleanup
free(params.weight);
free(params.bias);
tensor_free(input);
tensor_free(output);
```

## GTCRN Network Architecture

The GTCRN model from gtcrn1.py uses these operations in the following structure:

### Encoder
1. **ConvBlock**: Conv2d(9→16) + BatchNorm + PReLU
2. **ConvBlock**: Conv2d(16→16, groups=2) + BatchNorm + PReLU
3. **GTConvBlock**: Pointwise + Depthwise + Pointwise + TRA
4. **GTConvBlock**: Pointwise + Depthwise + Pointwise + TRA (dilation=2)
5. **GTConvBlock**: Pointwise + Depthwise + Pointwise + TRA (dilation=5)

### Bottleneck
- **DPGRNN**: Dual-path grouped RNN (2 layers)

### Decoder
1. **GTConvBlock**: Transpose version (dilation=5)
2. **GTConvBlock**: Transpose version (dilation=2)
3. **GTConvBlock**: Transpose version (dilation=1)
4. **ConvBlock**: ConvTranspose2d(16→16, groups=2) + BatchNorm + PReLU
5. **ConvBlock**: ConvTranspose2d(16→2) + BatchNorm + Tanh

## Implementation Notes

### Memory Layout
Tensors use NCHW format (batch, channels, height, width) with contiguous memory layout:
```
index = ((b * C + c) * H + h) * W + w
```

### Grouped Convolution
When `groups > 1`, the input and output channels are divided into groups:
- Input channels per group: `in_channels / groups`
- Output channels per group: `out_channels / groups`
- Each group processes independently

### Dilation
Dilation creates "atrous" (dilated) convolutions by inserting gaps between kernel elements:
```
effective_kernel_size = dilation * (kernel_size - 1) + 1
```

### Transpose Convolution
Output size calculation:
```
output_size = (input_size - 1) * stride - 2 * padding + kernel_size
```

## Performance Considerations

1. **Memory Access**: The implementation uses straightforward nested loops. For production use, consider:
   - Loop reordering for better cache locality
   - SIMD vectorization
   - Multi-threading for batch processing

2. **Optimization Opportunities**:
   - im2col + GEMM for regular convolutions
   - Winograd algorithm for small kernels
   - FFT-based convolution for large kernels

3. **Memory Management**:
   - Current implementation allocates tensors dynamically
   - Consider pre-allocating buffers for real-time applications

## Correspondence to PyTorch

| PyTorch Operation | C Function |
|------------------|------------|
| `nn.Conv2d` | `conv2d_forward` |
| `nn.ConvTranspose2d` | `conv2d_transpose_forward` |
| `nn.Conv2d(groups=in_channels)` | `depthwise_conv2d_forward` |
| `nn.Conv2d(kernel_size=1)` | `pointwise_conv2d_forward` |
| `nn.BatchNorm2d` | `batch_norm_2d_forward` |
| `nn.PReLU` | `prelu_forward` |
| `nn.Tanh` | `tanh_forward` |

## Testing

The test suite (`test_conv2d.c`) includes:

1. **Regular Conv2d Test**: 3→16 channels, 3x3 kernel
2. **Depthwise Conv2d Test**: 16 channels, 3x3 kernel
3. **Pointwise Conv2d Test**: 16→32 channels, 1x1 kernel
4. **Conv2d Transpose Test**: 16→8 channels, 4x4 kernel, stride=2
5. **BatchNorm + Activation Test**: Combined normalization and activation

Each test reports:
- Input/output tensor shapes
- Sample output values
- Execution time

## License

This implementation is provided for educational and research purposes.
