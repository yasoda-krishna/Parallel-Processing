/* 
 * File: omp_mm1.c
 * Purpose: Computes a parallel matrix-vector product with openMP
 * 
 * Linux Servers:
 * Compile: gcc -g -Wall -fopenmp -o omp_mm1 omp_mm1.c
 * Run: ./omp_mm1 <thread_count> <size of matrix>
 * 
 * Expanse Cluster:
 * 1. Load Intel compiler: module load intel mvapich2
 * 2. Compile code: icc -o mm1 omp_mm1.c -qopenmp
 * 3. Submit job script: sbatch mm1Script.sb
 * 
 * IT 388 - Illinois State University Homework 4
 * Name: Yasoda Krishna Reddy Annapureddy
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Function to print the matrix
void printMatrix(double *C, int m);

int main(int argc, char *argv[])
{
    double start, elapsed, finished;
    int i, j, k, nThreads;
    int m;
    double *A;
    double *B;
    double *C;
    

    // Get number of threads and matrix size from command line arguments
    if (argv[1] == 0 || argv[2] == 0)
    {
        fprintf(stderr, "\n \t USAGE: ./omp_mm <number of threads> <size of matrix> \n\n");
        exit(1);
    }
    nThreads = atoi(argv[1]);
    m = atoi(argv[2]);

    // Set number of threads to be used
    omp_set_num_threads(nThreads);

    // Allocate memory for matrix A, B, C, vector x, and y
    A = malloc(m * m * sizeof(double));
    B = malloc(m * m * sizeof(double));
    C = malloc(m * m * sizeof(double));

    // Generate matrix A and B
    #pragma omp parallel for private(i,j)
    for (i = 0; i < m; i++)
    {   
        # pragma omp parallel for
        for (j = 0; j < m; j++)
        {
            A[i * m + j] = (i%2) + (j%3);
            B[i * m + j] = (i%3) - j%2 ;
        }
    }

    // Measure initial time
    start = omp_get_wtime();

    // Matrix-matrix multiplication
    #pragma omp parallel for private(i,j,k)
    for(i = 0; i < m; i++)
    {
        for(j = 0; j < m; j++)
        {
            C[i*m+j] = 0;
            for (k = 0; k < m; ++k) 
            {
                C[i*m+j] += A[i*m+k] * B[k*m+j];
            }
        }
    }
    
    //Stop Measuring Time and Compute the Elapsed Time
    finished = omp_get_wtime();
    elapsed = finished - start;

    // Print elapsed time and matrix dimension
    printf("A[%d,%d] x B[%d,%d] = C[%d,%d], #threads: %d ,Elapsed time: %f sec\n", m, m, m, m, m, m, nThreads, elapsed);

    // Print matrix C if the size is less than 20
    if (m < 20)
    {
        printf("\nC=");
        printMatrix(C,m);
        printf("\n");
     }
    // Free allocated memory
    free(A);
    free(B);
    free(C);
    return 0;

}

// Function to print the matrix
void printMatrix(double *C, int m)
{
    int i, j;
    printf("\n[");
    for (i = 0; i < m; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            printf("%.f ", C[i * m + j]);
        }
        printf("]\n");
    }
}



