CUDA_HOME   = /Soft/cuda/8.0.61

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=compute_35 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
PROG_FLAGS  = -DSIZE=32


EXE4GPUs    = bitonic_1GPU.exe

OBJ4GPUs    = bitonic_1GPU.o

default: $(EXE4GPUs)

bitonic_1GPU.o: bitonic_1GPU.cu
	$(NVCC) -c -o $@ bitonic_1GPU.cu $(NVCC_FLAGS) $(PROG_FLAGS)


$(EXE4GPUs): $(OBJ4GPUs)
	$(NVCC) $(OBJ4GPUs) -o $(EXE4GPUs) $(LD_FLAGS)


all:	$(EXE4GPUs) 

clean:
	rm -rf *.o kernel*.exe

