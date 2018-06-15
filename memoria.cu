#include <stdio.h>
int main() {
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("%lu\n", prop.sharedMemPerBlock);	

	}
}
