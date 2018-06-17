#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <stdlib.h>

#define NUM_THREADS 1024
#define NUM_BLOCKS 32768
#define NUM_VALS NUM_THREADS*NUM_BLOCKS
#define SHARED_SIZE_LIMIT 1024

//Macro per a swap
#define SWAP(_i, _ixj){\
	int aux = vector[_i];\
	vector[_i] = vector[_ixj];\
	vector[_ixj] = aux;}

//Funcio de testeig per mirar que el vector esta ordenat
int testOrdenacio(int length, int *vector){
	int ordenat = 1;
	int i;
	for(i = 0; i < length -1 && ordenat; ++i){
		if(vector[i] > vector[i+1]) ordenat = 0;
	}
	return ordenat;
}

void array_copy(int *dst, int *src, int length) {
  	int i;
  	for (i=0; i<length; ++i) {
  		dst[i] = src[i];
  	}
}

//Comparamos dos elementos y en caso de ser decrecientes, los swapeamos.
__device__ inline void comparator(int &A, int &B, uint dir) {
    	int temp;
    	if ((A <= B) == dir) {
    	    	temp = A;
    	    	A = B;
    	    	B = temp;
    	}
}

__global__ void bitonicSortShared(int *dev_values){
   		int tx = threadIdx.x;
    	int bx = blockIdx.x;
    	int index = blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;

    	__shared__ int sh_values[SHARED_SIZE_LIMIT];
    	sh_values[tx] = dev_values[index];
    	sh_values[tx + (SHARED_SIZE_LIMIT/2)] = dev_values[index + (SHARED_SIZE_LIMIT/2)];
    	
		for (uint size = 2; size < SHARED_SIZE_LIMIT; size <<= 1) {
    	  	uint ddd = (tx & (size / 2)) == 0;//direction: ascending or descending
    	  	for (uint stride = size/2; stride > 0; stride >>= 1) {
    	  	    	__syncthreads();
    	  	    	uint pos = 2 * tx - (tx & (stride - 1));
    	  	    	comparator(sh_values[pos], sh_values[pos + stride], ddd);
    	  	}
    	}
    	
		uint ddd = ((bx&1) == 0); //    uint ddd = ((bx&1)==0);
    	{
    	  	for (uint stride = SHARED_SIZE_LIMIT/2; stride > 0; stride >>= 1) {
    	  	    	__syncthreads();
    	  	    	uint pos = 2 * tx - (tx & (stride - 1));
    	  	    	comparator(sh_values[pos + 0], sh_values[pos + stride], ddd);
    	  	}
    	}
    	
		__syncthreads();
    	dev_values[index] = sh_values[tx];
    	dev_values[index+(SHARED_SIZE_LIMIT/2)] = sh_values[tx+(SHARED_SIZE_LIMIT/2)];
}

void bitonic_sort(int *values){
  	int *dev_values;
  	size_t size = NUM_VALS * sizeof(int);
  	
	cudaMalloc((void**) &dev_values, size);
  	cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  	
	dim3 numBlocks(NUM_BLOCKS, 1);
  	dim3 numThreads(NUM_THREADS, 1);
  	
	cudaDeviceSynchronize();
  	
	uint blockCount = NUM_VALS / SHARED_SIZE_LIMIT;
  	uint threadCount = SHARED_SIZE_LIMIT / 2;
  	printf("blockCount=%d, threadCount=%d, SHARED_SIZE_LIMIT=%d\n", blockCount, threadCount, SHARED_SIZE_LIMIT);
  	
	bitonicSortShared<<<blockCount, threadCount>>>(dev_values);
  	
	cudaDeviceSynchronize();
  	
	cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  	cudaFree(dev_values);
}

int main(void){
  	int *host_values;
  	cudaMallocHost( &host_values, NUM_VALS * sizeof(int));

  	float TiempoKernel; 

  	cudaEvent_t E1, E2;

  	cudaEventCreate(&E1);
  	cudaEventCreate(&E2);

  	//Inicialitzem vector amb valors random 
  	int i;
  	srand(time(NULL));
  	for(i = 0; i < NUM_VALUES; ++i){
  		host_v[i] = rand();
  	}

  	cudaEventRecord(E1, 0);
  	cudaEventSynchronize(E1);

  	cudaFuncSetCacheConfig(bitonicSortShared, cudaFuncCachePreferL1);

  	bitonic_sort(host_values);
  	
	testOrdenacio(NUM_VALS, host_values);	

	cudaEventRecord(E2, 0);
  	cudaEventSynchronize(E2);

  	cudaEventElapsedTime(&TiempoKernel, E1, E2);

  	printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);

  	cudaFree(host_values);
}
