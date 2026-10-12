DTK_ROOT ?= /opt/dtk-25.04.4
HIPCC ?= $(DTK_ROOT)/bin/hipcc
GPU_ARCH ?= gfx936
CB_BLOCK_SIZE ?= 16
CB_WAVES_PER_BLOCK ?= 4
CB_ROW_OWNED ?= 1
CB_PATTERN_WAVE ?= 1

TARGET := cb-spmv-dcu
HEADERS := $(wildcard *.h *.hpp io/*.h)
CXXFLAGS := -O3 -ffast-math -w -std=c++17 -fopenmp --offload-arch=$(GPU_ARCH) \
	--gpu-max-threads-per-block=1024 -DCB_BLOCK_SIZE=$(CB_BLOCK_SIZE) \
	-DCB_WAVES_PER_BLOCK=$(CB_WAVES_PER_BLOCK) \
	-DCB_ROW_OWNED=$(CB_ROW_OWNED) \
	-DCB_PATTERN_WAVE=$(CB_PATTERN_WAVE)
LDFLAGS := -fopenmp -Wl,-rpath,$(DTK_ROOT)/lib \
	-Wl,-rpath,$(DTK_ROOT)/hip/lib

.PHONY: all clean

all: $(TARGET)

$(TARGET): main.hip.cpp $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	$(RM) $(TARGET)
