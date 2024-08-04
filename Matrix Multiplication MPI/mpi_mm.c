/* File:     mpi_mm.c
 *
 * Name: Yasoda Krishna Reddy Annapureddy
 * Course: IT388 Parallel Processing
 * Homework: 3 
 *
 *
 * Compile:  mpicc -g -Wall -o mpi_mm mpi_mm.c
 * Run:      mpiexec -n <number of processes> ./mpi_mm <M>
 *              M is the dimension of the Matrix A
 *              
 * Note: Assume M divisible by the number of processors.
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void printMatrix(double *y,int M);

int main(int argc, char* argv[]) {
    double* local_A;
    double* local_C;
    double* A;
    double* B;
    double* C;
    int my_rank, nproc;
    MPI_Comm comm;
    int N, M ,Q ,localN, localQ , localM;
    double start, elapsed;
    
    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &my_rank);

    if (my_rank==0){
        if (argv[1]==0 || argv[2]==0){
            fprintf(stderr,"\n---USAGE: mpiexec -n <#proc> ./mpi_mm < #matrix_dimension>  \n\n");
            MPI_Abort(comm,1);
        }
        N = atoi(argv[1]);
        M = atoi(argv[2]);
       Q = atoi(argv[3]);


    }
    
    MPI_Bcast(&N, 1, MPI_INT, 0, comm);
    MPI_Bcast(&)

    if(N%nproc!=0){
        fprintf(stderr,"\n---ERROR: N should be multiple of the number of processors \n\n");
        MPI_Abort(comm,1);
    }
    
    localM = M/nproc;
    
    // Allocate matrix - Memory allocation very important
    A = malloc(M*M*sizeof(double));
    B = malloc(M*M*sizeof(double));
    C = malloc(M*M*sizeof(double));
    local_A=malloc(M*M*sizeof(double));
    local_C=malloc(localM*M*sizeof(double));
    
    
    // Generate matrix A and B
    if (my_rank==0){
        int i, j;
        for ( i = 0; i < M; ++i) {
            for ( j = 0; j < M; ++j) {
                A[i*M+j] = (i%2) + j%3;
                B[i*M+j] = (i%3) - j%2;
            }
        }
    }
    
    MPI_Barrier(comm);
    start = MPI_Wtime();
    
    //1. Broadcast Matrix B to all processors
    MPI_Bcast(B, M*M, MPI_DOUBLE, 0, comm);
    
    //2. Distribute a row-block of matrix A to each processor
    MPI_Scatter(A, localM*M, MPI_DOUBLE, local_A, localM*M, MPI_DOUBLE, 0, comm);
    
    //3. Each processor computes its multiplication
    int local_i=0, local_j=0, local_k=0;
    for ( local_i = 0; local_i < localM; local_i++) {
        for ( local_j = 0; local_j < M; local_j++) {
            local_C[local_i*M+local_j]=0;
            for ( local_k = 0; local_k < M; local_k++) {
                local_C[local_i*M+local_j] += local_A[local_i*M+local_k] * B[local_k*M+local_j];
            }
        }
    }

    // 4. Gather local_C calculations of matrix-matrix multiplication
    MPI_Gather(local_C, M*localM, MPI_DOUBLE, C, M*localM, MPI_DOUBLE, 0, comm);
 
    
    
    MPI_Barrier(comm);
    elapsed = MPI_Wtime()-start;
    
    // Print Elapsed time and final matrix if M is less than 20
    if (my_rank == 0) {
        if(M<20){
            printf ("\n Matrix A \n");
            printMatrix(A,M);
            printf ("\n Matrix B \n");
            printMatrix(B,M);
        }
        printf ("\n Elapsed time %f sec (Parallel) \n \n",elapsed);
        if(M<20){
            printf("\n Matrix Multiplication of A and B of dimension : %d \n",M);
            printMatrix(C,M);
        }
    }
    
    MPI_Finalize();
    return 0;
}  /* main */

/** Print Matrix to screen */
void printMatrix(double *C,int N){
    int i,j;
    for ( i = 0; i < N; ++i) {
        for ( j = 0; j < N; ++j) {
            printf("%.f ",C[i*N+j]);
        }
        printf("\n");
    }
}