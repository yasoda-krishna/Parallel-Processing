/* File:   hy_mv.c
 * Purpose:  Parallel matrix-vector multiplication using
 *           one-dimensional arrays to store the vectors and the
 *           matrix.  The matrix is distributed by block rows.
 *           Vectors are broadcast to all processes.
 * Compile:  mpicc -g -Wall -o mv hy_mv.c -fopenmp
 * Run:      mpiexec -n <number of processes> ./mv <N>
 *              N is the size of the matrix
 * Input:    Number of rows and number of columns (square matrix)
 * Output:   Elapsed time
 * Notes:
 *    1. Number of processes should evenly divide N
 *
 * TODO: Implement openMP parallelization
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>


/*-------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
    double* local_A;
    double* local_y;
    double* x;
    double* y;
    double* A;
    int my_rank, nproc;
    MPI_Comm comm;
    int m, local_m, n;
    double start, elapsed;
	int nThreads = 0;
    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &my_rank);

    // Get dimesnions for matrix and vector
    if (my_rank==0){
        m=n=atoi(argv[1]);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    MPI_Bcast(&m, 1, MPI_INT, 0, comm);

    local_m = m/nproc;

    // Allocate vector and matrix
    y = malloc(m*sizeof(double));
    A = malloc(m*n*sizeof(double));
    x = malloc(n*sizeof(double));
    local_y=malloc(local_m*sizeof(double));
    local_A=malloc(local_m*n*sizeof(double));


    // Generate matrix and vector
    if (my_rank==0){
        printf("\n Matrix size: %dx%d  \n",m,n);
        int i, j;
        for (i=0;i<m;i++){
            for (j=0;j<n;j++){
                A[i*n + j] = rand()%6;
            }
        }
        // Generate vector x
        int k;
        for (k=0;k<n;k++){
            x[k] = rand()%6;
        }
    }

    MPI_Barrier(comm);
    start=MPI_Wtime();
    // Broadcast x to all processes
    MPI_Bcast(x, n, MPI_DOUBLE, 0, comm);

    // Distribute a row-block to each process
    MPI_Scatter(A, local_m*n, MPI_DOUBLE,
                local_A, local_m*n, MPI_DOUBLE, 0, comm);

    // Each process computes its row block of Matrix-vector multiplication
    int local_i=0, j=0;
    // Add openMP construts
#pragma omp parallel
{
#pragma omp_single
nThreads = omp_get_num_threads();
#pragma omp parallel for private(local_i,j)
    for (local_i = 0; local_i < local_m; local_i++) {
        local_y[local_i] = 0.0;
        for (j = 0; j < n; j++){
            local_y[local_i] += local_A[local_i*n+j]*x[j];
        }
    }
}
    MPI_Allgather(local_y, local_m, MPI_DOUBLE, y, local_m, MPI_DOUBLE, comm);

    MPI_Barrier(comm);
    elapsed = MPI_Wtime()-start;

    // Update the priting statement to print: Matrix dimension, number processors, number threads
    if (my_rank == 0) {
        printf("\n Matrix dimensions are [%d,%d] ,No of Processors : %d, No of Threads : %d , Elapsed %lf sec\n",m,m,nproc,nThreads,elapsed);
    }

    MPI_Finalize();
    return 0;
}  /* main */
