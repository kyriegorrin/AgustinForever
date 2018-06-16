#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>

//Afegim un tamany per defecte
//Imprescindible que sigui potencia de 2
#define NUM_THREADS 1024
#define NUM_BLOCKS 32768
#define N NUM_THREADS*NUM_BLOCKS

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
		if((i & k) == 0 && vector[i] > vector[ixj]){
			SWAP(i, ixj);
			//int aux = vector[i];
			//vector[i] = vector[ixj];
			//vector[ixj] = aux;
		}
		if((i & k) != 0 && vector[i] < vector[ixj]){
			SWAP(i, ixj);
			//int aux = vector[i];
			//vector[i] = vector[ixj];
			//vector[ixj] = aux;
		}
	}
}

//Funcio iterativa de bitonic sort
void bitonicSort(int *vector){
	int j, k;

	dim3 numBlocks(NUM_BLOCKS, 1);
	dim3 numThreads(NUM_THREADS, 1);

	for(k = 2; k <= N; k = 2*k){
		//Els shifts son equivalents de dividir entre 2
		for(j = k >> 1; j > 0; j = j >> 1){ 
			bitonicSortKernel<<<numBlocks, numThreads>>>(vector, j, k);
		}
	}
}

int testOrdenacio(int *vector){
	int ordenat = 1;
	int i;
	for(i = 0; i < N -1 && ordenat; ++i){
		if(vector[i] > vector[i+1]) ordenat = 0;
	}
	return ordenat;
}

int main(int argc, char **argv) {
	//Generacio dels parametres del vector
	int n = N;
	//if(argc > 1) n = atoi(argv[1]); 
	int *vector, *vectorDevice, *vectorAux;

	cudaEvent_t E0, E1, E2, E3;

	cudaEventCreate(&E0);
	cudaEventCreate(&E1);
	cudaEventCreate(&E2);
	cudaEventCreate(&E3);
	
	unsigned int NUM_BYTES = N*sizeof(int);

	//Reserva de memoria per als vectors
	cudaMallocHost(&vector, NUM_BYTES);
	cudaMallocHost(&vectorAux, NUM_BYTES);
	
	//Inicialitzacio amb valors random
	int i;
	srand(time(NULL));
	for(i = 0; i < N; ++i){
		vector[i] = rand();
	}

	cudaEventRecord(E0, 0);
	cudaEventSynchronize(E0);
	
	cudaMalloc((int **)&vectorDevice, NUM_BYTES);

	cudaEventRecord(E1, 0);
	cudaEventSynchronize(E1);

	//Pas del vector de host a device
	cudaMemcpy(vectorDevice, vector, NUM_BYTES, cudaMemcpyHostToDevice);

	//Fem sort del vector
	bitonicSort(vector);

	cudaEventRecord(E2, 0);
	cudaEventSynchronize(E2);

	//Pas del vector de device a host
	cudaMemcpy(vector, vectorDevice, NUM_BYTES, cudaMemcpyDeviceToHost);

	//Test per veure si la ordenacio es correcte
	if(testOrdenacio(vector)) printf("Agustin is happy\n");
	else printf("Agustin te deniega el curso PUMPS\n");

	//Alliberacio de memoria
	cudaFree(vector);
	cudaFree(vectorDevice);

	cudaDeviceSynchronize();
	cudaEventRecord(E3, 0);
	cudaEventSynchronize(E3);

	//Timing
	float tempsTotal;
	cudaEventElapsedTime(&tempsTotal, E0, E3);
	
	printf("Temps: %f", tempsTotal);

	//Destrueix events
	cudaEventDestroy(E0);
	cudaEventDestroy(E1);
	cudaEventDestroy(E2);
	cudaEventDestroy(E3);

}


