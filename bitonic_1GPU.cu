#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>

#define NUM_THREADS     1024				 
#define NUM_BLOCKS 	1024	
#define NUM_VALUES NUM_THREADS*NUM_BLOCKS

//Macro per a swap
#define SWAP(_i, _ixj){\
	int aux = vector[_i];\
	vector[_i] = vector[_ixj];\
	vector[_ixj] = aux;}

//Kernel per a bitonic sort
__global__ void bitonicSortKernel(int *vector, int j, int k){
	int i, ixj;
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i ^ j;

	if((ixj) > i){
		if((i & k) == 0 && vector[i] > vector[ixj])
			SWAP(i, ixj);
		if((i & k) != 0 && vector[i] < vector[ixj])
			SWAP(i, ixj);
	}
}

//Funcio iterativa de bitonic sort
void bitonicSort(int length, int *vector){
	int j, k;	

	dim3 numBlocks(NUM_BLOCKS, 1);
	dim3 numThreads(NUM_THREADS, 1);

	for(k = 2; k <= length; k = 2*k){
		//Els shifts son equivalents de dividir entre 2
		for(j = k >> 1; j > 0; j = j >> 1){ 
			bitonicSortKernel<<<numBlocks, numThreads>>>(vector, j, k);
		}
	}
}

//Funcio de testeig per mirar que el vector esta ordenat
int testOrdenacio(int length, int *vector){
	int ordenat = 1;
	int i;
	for(i = 0; i < length -1 && ordenat; ++i){
		if(vector[i] > vector[i+1]) ordenat = 0;
	}
	return ordenat;
}

int main(int argc, char **argv){
	int n = NUM_VALUES;
	if(argc > 1) n = atoi(argv[1]); 

	//Vectors i variables auxiliars	
  	int *host_v, *dev_v;

	//Creacio d'events
  	cudaEvent_t E0, E1, E2, E3;
  	cudaEventCreate(&E0);
  	cudaEventCreate(&E1);  
  	cudaEventCreate(&E2);  
  	cudaEventCreate(&E3);  

  	unsigned int numBytes = NUM_VALUES * sizeof(int);

	//Timing
  	cudaEventRecord(E0, 0);
  	cudaEventSynchronize(E0);

	//Reservem memoria al host
  	cudaMallocHost( &host_v, numBytes);

  	//Inicialitzem vector amb valors random 
  	int i;
  	srand(time(NULL));
  	for(i = 0; i < NUM_VALUES; ++i){
  		host_v[i] = rand();
  	}
	
  	//Reservem memoria al device
  	cudaMalloc((int**)&dev_v, numBytes);

  	//Enviem les dades del host al device
  	cudaMemcpy(dev_v, host_v, numBytes, cudaMemcpyHostToDevice);

	//Timing	
  	cudaEventRecord(E2, 0);
  	cudaEventSynchronize(E2);

  	//Executem el kernel
  	bitonicSort(NUM_VALUES ,dev_v);

	//Timing	
  	cudaEventRecord(E3, 0);
  	cudaEventSynchronize(E3);

  	//Recuperem les dades tractades del device
  	cudaMemcpy( host_v, dev_v, numBytes, cudaMemcpyDeviceToHost);

  	//Apliquem el test de correctesa
  	if(testOrdenacio(NUM_VALUES, host_v)) printf("TEST CORRECTO\n");
  	else printf("TEST FALLADO\n\n");
  	
  	//Alliberem memoria reservada anteriorment
  	cudaFree(dev_v);
  	cudaFree(host_v); 

	//Timing
  	cudaDeviceSynchronize();
  	cudaEventRecord(E1, 0);
  	cudaEventSynchronize(E1);

	//Calculem i mostrem temps d'execucio del programa
  	float tempsTotal, tempsKernel; 
  	cudaEventElapsedTime(&tempsTotal, E0, E1);
  	cudaEventElapsedTime(&tempsKernel, E2, E3);

	n = NUM_VALUES;

  	printf("Numero de threads: %d\n", NUM_THREADS);
  	printf("Numero de blocks: %d\n", NUM_BLOCKS);
	printf("Numero de valores de entrada: %d\n", n);
  	printf("Tiempo total de programa: %f ms\n", tempsTotal);
	printf("Tiempo de kernel: %f ms\n", tempsKernel);

	//Destruim els events
  	cudaEventDestroy(E0); 
  	cudaEventDestroy(E1);
	cudaEventDestroy(E2);
	cudaEventDestroy(E3);
} 

