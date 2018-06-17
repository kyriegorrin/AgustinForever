#include <stdio.h>
#include <time.h>
#include <stdlib.h>

//Afegim un tamany per defecte
#define N 16777216

//Funció de canvi entre valors de posicions de vector
//Assumim que el valors "i" i "j" estan dins del seu rang
void swap(int *vector, int i, int j){
	int aux = vector[i];
	vector[i] = vector[j];
	vector[j] = aux;
}

//Funcio iterativa de bitonic sort
void bitonicSort(int length, int *vector){
	int i, j, k;
	for(k = 2; k <= length; k = 2*k){
		//Els shifts son equivalents de dividir entre 2
		for(j = k >> 1; j > 0; j = j >> 1){ 
			for(i = 0; i < length; ++i){
				int ixj = i ^ j;
				if((ixj) > i){
					if((i & k) == 0 && vector[i] > vector[ixj]) swap(vector, i, ixj);
					if((i & k) != 0 && vector[i] < vector[ixj]) swap(vector, i, ixj);
				}
			}
		}
	}
}

int testOrdenacio(int length, int *vector){
	int ordenat = 1;
	int i;
	for(i = 0; i < length -1 && ordenat; ++i){
		if(vector[i] > vector[i+1]) ordenat = 0;
	}
	return ordenat;
}

int main(int argc, char **argv) {
	//Generacio dels parametres del vector
	int n = N;
	if(argc > 1) n = atoi(argv[1]); 
	int vector[n];

	//Agafem el temps d'inici de calculs
	clock_t begin = clock();
	
	//Inicialitzacio amb valors random
	int i;
	srand(time(NULL));
	for(i = 0; i < n; ++i){
		srand(i);
		vector[i] = rand();
	}
	
	//Fem sort del vector
	clock_t beginSort = clock();
	bitonicSort(n, vector);
	clock_t endSort = clock();

	//Test per veure si la ordenacio es correcte
	if(testOrdenacio(n, vector)) printf("TEST CORRECTO\n");
	else printf("TEST FALLADO\n\n");

	//Agafem el temps final de calculs
	clock_t end = clock();

	//Printem els temps obtinguts
	double tempsTotal = (double)(end - begin) / CLOCKS_PER_SEC;
	double tempsKernel = (double) (endSort - beginSort) / CLOCKS_PER_SEC;

	printf("Numero de valores de entrada: %d\n", n);
	printf("Tiempo total de programa: %f segundos\n", tempsTotal);
	printf("Tiempo de kernel: %f segundos\n", tempsKernel);

	return 0;
}


