#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>

#define NUM_THREADS     1024				 
#define NUM_BLOCKS 	32768			
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

int main(){
	//Vectors i variables auxiliars	
  	int *host_v, *dev_v;
  	float tempsTotal; 

	//Creacio d'events
  	cudaEvent_t E0, E1;
  	cudaEventCreate(&E0);
  	cudaEventCreate(&E1);  

  	unsigned int numBytes = NUM_VALUES * sizeof(int);

	//Reservem memoria al host
  	cudaMallocHost( &host_v, numBytes);

  	//Inicialitzem vector amb valors random 
  	int i;
  	srand(time(NULL));
  	for(i = 0; i < NUM_VALUES; ++i){
  		host_v[i] = rand();
  	}

  	cudaEventRecord(E0, 0);
  	cudaEventSynchronize(E0);

  	//Reservem memoria al device
  	cudaMalloc((int**)&dev_v, numBytes);

  	//Enviem les dades del host al device
  	cudaMemcpy(dev_v, host_v, numBytes, cudaMemcpyHostToDevice);

  	//Executem el kernel
  	bitonicSort(NUM_VALUES ,dev_v);

  	//Recuperem les dades tractades del device
  	cudaMemcpy( host_v, dev_v, numBytes, cudaMemcpyDeviceToHost);

  	//Apliquem el test de correctesa
  	if(testOrdenacio(NUM_VALUES, host_v)) printf("Agustin is happy\n");
  	else printf("Agustin te deniega el curso PUMPS\n");
  	
  	//Alliberem memoria reservada anteriorment
  	cudaFree(dev_v);
  	cudaFree(host_v); 

  	cudaDeviceSynchronize();
  	cudaEventRecord(E1, 0);
  	cudaEventSynchronize(E1);

	//Calculem i mostrem temps d'execucio del programa
  	cudaEventElapsedTime(&tempsTotal, E0, E1);

  	printf("Nombre de threads utilitzat: %d\n", NUM_THREADS);
  	printf("Nombre de blocks utilitzat: %d\n", NUM_BLOCKS);
  	printf("Temps total: %4.6f milseg\n", tempsTotal);

	//Destruim els events
  	cudaEventDestroy(E0); 
  	cudaEventDestroy(E1);
} 

