DTK_ROOT ?= /opt/dtk-25.04.4
HIPCC ?= $(DTK_ROOT)/bin/hipcc
GPU_ARCH ?= gfx936

TARGET := hipsparse_csr_spmv
HEADERS := $(wildcard *.h io/*.h)
CXXFLAGS := -O3 -w -std=c++17 --offload-arch=$(GPU_ARCH) \
	--gpu-max-threads-per-block=512 -I$(DTK_ROOT)/include
LDFLAGS := -L$(DTK_ROOT)/lib -lhipsparse \
	-Wl,-rpath,$(DTK_ROOT)/lib -Wl,-rpath,$(DTK_ROOT)/hip/lib

.PHONY: all clean

all: $(TARGET)

$(TARGET): main.hip.cpp $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	$(RM) $(TARGET)
