/* File: mpi_array.c
 Processing array using point-to-point communication.

 --Solution---
 Name: Rosangela Follmann
 IT 388 - Spring 2023
 */
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
void printArray(int[], int size);

int main(int argc, char *argv[])
{

    /* global variables*/
    int N; // array lenght
    int *array;
    int *array_new;

    int rank, nproc;
    int offset, n_local, dest, source;
    double startTime, elapsedTime;
    /* Start MPI*/
    MPI_Init(&argc, &argv);
    MPI_Comm comm;
    MPI_Status status;
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);
    // Get the name of the processor
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(proc_name, &name_len);

    MPI_Barrier(comm);
    startTime = MPI_Wtime();
    if (rank == 0)
    {                  // Manager get User Input
        if (argc == 1) // no argument provided, abort
        {
            fprintf(stderr, "\n---USAGE: mpiexec -n <#proc> %s <data Input> \n\n", argv[0]);
            MPI_Abort(comm, 1);
        }
        N = atoi(argv[1]);
        /* manager sends N to all other processes */
        for (dest = 1; dest < nproc; dest++)
        {
            MPI_Send(&N, 1, MPI_INT, dest, 10, comm); // actual subarray data
        }
    }
    else
    {
        // workers receive problem size N
        MPI_Recv(&N, 1, MPI_INT, 0, 10, comm, &status);
    }

    // Allocate memory for arrays
    array = malloc(N * sizeof(int));
    array_new = malloc(N * sizeof(int));
    n_local = N / nproc;

    if (rank == 0)
    {
        // Manager generates array
        for (int i = 0; i < N; i++)
        {
            array[i] = i * 2 + 1;
        }
        if (N < 20)
        {
            printf("Inital array = ");
            printArray(array, N);
        }
        /* 2.3. manager sends subarrays to each process */
        for (dest = 1; dest < nproc; dest++)
        {
            offset = n_local * dest;
            MPI_Send(&array[offset], n_local, MPI_INT, dest, 22, comm); // actual subarray data
        }
        /* 2.4. manager works on its part of the array, and stores it in new_array */
        for (int j = 0; j < n_local; j++)
        {
            array_new[j] = array[j] * 10;
        }
        /* 2.5.  manager receives calculations from workers */
        for (source = 1; source < nproc; source++)
        {
            offset = source * n_local;
            MPI_Recv(&array_new[offset], n_local, MPI_INT, source, 33, comm, &status);
        }
        /* 2.6. manager prints the processed array */
        if (N < 20)
        {
            printf("Final array = ");
            printArray(array_new, N);
        }
    }
    else
    { // Workers

        /* 3.0. Workers receive offset or compute their own, and its portion of the data */
        offset = n_local * rank;
        MPI_Recv(&array[offset], n_local, MPI_INT, 0, 22, comm, &status);
        /* 3.1 Worker prints processor name, its rank and received array*/
        if (N < 20)
        {
            printf("%s : rank %d: received subarray ", proc_name, rank);
            printArray(array, N);
        }
        /* 3.2. Workers works on their part of the array: multiply each element by 10, and updates new_array. */
        for (int j = offset; j < offset + n_local; j++)
        {
            array_new[j] = array[j] * 10;
        }
        /* 3.3. Workers send their work back to manager */
        MPI_Send(&array_new[offset], n_local, MPI_INT, 0, 33, comm);
    }
    MPI_Barrier(comm);
    elapsedTime = MPI_Wtime() - startTime;

    if (rank == 0)
    {

        printf("Point-to-Point: %s, #proc = %d, N = %d,  elapsed time %lf msec \n ", proc_name, nproc, N, elapsedTime * 1000);
    }

    MPI_Finalize();
    return 0;
}
/*--------------------
 Print array to screen
 */
void printArray(int a[], int N)
{
    if (N < 20)
    {
        printf("[");
        for (int i = 0; i < N - 1; i++)
        {
            printf("%d,", a[i]);
        }
        printf("%d]\n", a[N - 1]);
    }
}
