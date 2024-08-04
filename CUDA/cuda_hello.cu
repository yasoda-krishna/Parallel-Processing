/* FILE: cuda_hello.cu
Compile and run in Expanse cluster
1) Load module:
	module load cuda
2) Compile
nvcc -o cuda_hello cuda_hello.cu

3) a) Submit job
	sbatch jobCuda_hello.jb
*/

#include<stdio.h>
#include<stdlib.h>

// Kernel function
__global__ void hello() {
	printf("Device: Hello World! from thread [ %d, %d]\n",blockIdx.x,threadIdx.x);
}

_
int main(void) {
	printf("Hello World from host!\n");
	// Lauching kernel function<<<nBlocks,nThreads>>>(arguments)
	hello<<<2,4>>>();
	cudaDeviceSynchronize();
return 0;
}
