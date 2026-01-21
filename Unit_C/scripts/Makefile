# Makefile for GTCRN C Implementation

CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -ffast-math -std=c99
LDFLAGS = -lm
DEBUG_FLAGS = -g -O0 -DDEBUG

# Source files
CORE_SRCS = gtcrn_model.c gtcrn_modules.c gtcrn_streaming.c gtcrn_streaming_optimized.c
LAYER_SRCS = conv2d.c batchnorm2d.c nn_layers.c layernorm.c GRU.c
SIGNAL_SRCS = stft.c gtconvblock_forward_complete.c
UTIL_SRCS = weight_loader.c

ALL_SRCS = $(CORE_SRCS) $(LAYER_SRCS) $(SIGNAL_SRCS) $(UTIL_SRCS)
ALL_OBJS = $(ALL_SRCS:.c=.o)

# Targets
TARGETS = denoise test_gtcrn test_stft test_gru test_conv2d
COMPLETE_TARGETS = example_complete

.PHONY: all clean debug test install export_weights help complete

all: $(TARGETS)

complete: $(COMPLETE_TARGETS)

# Main executable
denoise: example_realtime_denoise.c $(ALL_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ Built denoise executable"

# Test executables
test_gtcrn: test_gtcrn_model.c $(ALL_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ Built test_gtcrn"

test_stft: test_stft.c stft.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ Built test_stft"

test_gru: test_gru.c GRU.o nn_layers.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ Built test_gru"

test_conv2d: test_conv2d.c conv2d.o batchnorm2d.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ Built test_conv2d"

# Complete modules example
example_complete: example_use_complete_modules.c $(ALL_OBJS) \
                  GRU_bidirectional_complete.c gtconvblock_forward_complete.c
	$(CC) $(CFLAGS) -o $@ $< $(ALL_OBJS) $(LDFLAGS)
	@echo "✓ Built example_complete (with complete implementations)"

# Object files
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build
debug: CFLAGS = -Wall -Wextra $(DEBUG_FLAGS) -std=c99
debug: clean all
	@echo "✓ Built debug version"

# Run tests
test: test_stft test_gru test_conv2d
	@echo "Running tests..."
	@echo "================================"
	@echo "Testing STFT..."
	./test_stft
	@echo "================================"
	@echo "Testing GRU..."
	./test_gru
	@echo "================================"
	@echo "Testing Conv2D..."
	./test_conv2d
	@echo "================================"
	@echo "✓ All tests passed"

# Export weights from PyTorch model
export_weights:
	@echo "Exporting GTCRN weights from PyTorch model..."
	python3 ../export_weights.py --model ../checkpoints/model_trained_on_dns3.pth --output weights/
	@echo "✓ Weights exported to weights/"

# Install (copy to /usr/local/bin)
install: denoise
	@echo "Installing to /usr/local/bin..."
	sudo cp denoise /usr/local/bin/
	@echo "✓ Installed successfully"

# Clean
clean:
	rm -f $(ALL_OBJS) $(TARGETS) $(COMPLETE_TARGETS) *.o
	@echo "✓ Cleaned build files"

# Clean everything including weights
distclean: clean
	rm -rf weights/
	@echo "✓ Deep clean complete"

# Help
help:
	@echo "GTCRN C Implementation - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all executables (default)"
	@echo "  complete       - Build examples with complete implementations"
	@echo "  denoise        - Build main denoising executable"
	@echo "  test           - Build and run all tests"
	@echo "  debug          - Build with debug symbols"
	@echo "  export_weights - Export weights from PyTorch model"
	@echo "  clean          - Remove build files"
	@echo "  distclean      - Remove build files and weights"
	@echo "  install        - Install to /usr/local/bin"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make                # Build all"
	@echo "  make complete       # Build complete module examples"
	@echo "  make test           # Run tests"
	@echo "  make export_weights # Export weights"
	@echo "  make clean          # Clean"
	@echo "  make debug          # Debug build"

# Dependencies
gtcrn_model.o: gtcrn_model.c gtcrn_model.h gtcrn_modules.h conv2d.h batchnorm2d.h nn_layers.h GRU.h
gtcrn_modules.o: gtcrn_modules.c gtcrn_modules.h
gtcrn_streaming.o: gtcrn_streaming.c gtcrn_streaming.h gtcrn_model.h stft.h
gtcrn_streaming_optimized.o: gtcrn_streaming_optimized.c gtcrn_streaming.h gtcrn_model.h
conv2d.o: conv2d.c conv2d.h
batchnorm2d.o: batchnorm2d.c batchnorm2d.h
nn_layers.o: nn_layers.c nn_layers.h
layernorm.o: layernorm.c layernorm.h
GRU.o: GRU.c GRU.h
stft.o: stft.c stft.h
weight_loader.o: weight_loader.c weight_loader.h
gtconvblock_forward_complete.o: gtconvblock_forward_complete.c gtcrn_model.h
