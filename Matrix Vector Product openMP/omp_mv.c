/* File: omp_mv.c
 * Purpose: Computes a parallel matrix-vector product with openMP
 * Linux Servers:
 *  Compile:  gcc -g -Wall -fopenmp -o omp_mv omp_mv.c
 *  Run: ./omp_mv <thread_count> <rows> <columns>
 * Expanse Cluster:
 *  1) load intel compiler
        module load intel mvapich2
    2) compile code with
        icc -o mv omp_mv.c -qopenmp
    3) submit job script:
        sbatch ompScript.sb
 *
 * IT 388 - Illinois State University
 Name:

 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
void printArray(long *y, int M);
void printMatrix(long *C, int numRows, int numCol);
/*------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    double start, elapsed;
    int i, j, nThreads;
    int m, n;
    long *A;
    long *x;
    long *y;

    /* Get number of threads from command line */
    if (argv[1] == 0 || argv[2] == 0 || argv[3] == 0)
    {
        fprintf(stderr, "\n \t USAGE: ./omp_mv <number of threads> <rows> <columns> \n\n");
        exit(1);
    }
    nThreads = atoi(argv[1]);
    m = atoi(argv[2]);
    n = atoi(argv[3]);
    omp_set_num_threads(nThreads);

    A = malloc(m * n * sizeof(long));
    x = malloc(n * sizeof(long));
    y = malloc(m * sizeof(long));

    // 1. Mesure initial time
    start = omp_get_wtime();

    // 2. Generate matrix A
#pragma omp parallel for private(j)
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            A[i * n + j] = j % 6;
        }
    }
    if (m < 20 && n < 20)
    {
        printf("A=");
        printMatrix(A, m, n);
    }
    // 3. Generate vector x
#pragma omp parallel for
    for (i = 0; i < n; i++)
    {
        x[i] = i % 6;
    }
    if (n < 20)
    {
        printf("x=");
        printArray(x, n);
    }
// 4. Matrix-vector multiplication
#pragma omp parallel for private(i, j)
    for (i = 0; i < m; i++)
    {
        y[i] = 0.0;
#pragma omp parallel for
        for (j = 0; j < n; j++)
        {
            y[i] += A[i * n + j] * x[j];
        }
    }

    // 5. Mesure elapsed time

    elapsed = omp_get_wtime() - start;

    // 6. Print elapsed time and matrix dimension
    printf("A[%d,%d] x[%d], #threads: %d ,Elapsed time: %f\n", m, n, n, nThreads, elapsed);
    if (m < 20 && n < 20)
    {
        printf("y=");
        printArray(y, m);
    }
    free(A);
    free(x);
    free(y);
    return 0;
} /* main */

/** Print Matrix to screen */
void printArray(long *y, int M)
{
    int i, j;
    printf("[");
    for (i = 0; i < M; ++i)
    {
        printf("%ld ", y[i]);
    }
    printf("]\n");
}

void printMatrix(long *C, int numRows, int numCol)
{
    int i, j;
    printf("[");
    for (i = 0; i < numRows; ++i)
    {
        for (j = 0; j < numCol; ++j)
        {
            printf("%ld ", C[i * numCol + j]);
        }
        printf("]\n");
    }
}
