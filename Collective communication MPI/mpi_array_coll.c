/* File: mpi_array_coll.c
 * Process an array using collective communication in mpi.
 IT 388 - Spring 2023
 
 Your Name: Yasoda Krishna Reddy Annapureddy
 Date: 02/14/2023
 
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void printArray(int [],int size);

int main(int argc, char* argv[]){
    int N, n_local, rank, nproc;;
    int *array;
    int *array_new;
    MPI_Init(&argc,&argv);
    MPI_Comm comm; comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);
    
    // 1. Manager processor gets problem size from user command line input, allocate memory for both arrays, initializes array with odd numbers, computes n_local (work), and prints initial array
    if(rank==0){	
	N = atoi(argv[1]);    
	//printf("Enter the Size of Array : ");
        //scanf("%d",&N);

        array=malloc(N*sizeof(int));
        array_new=malloc(N*sizeof(int));

        n_local=N/nproc;

        for(int j=0;j<N;j++){
            array[j]=j*2 +1;
        }

        printf("Intial Array = "); printArray(array,N);

    }
 
    
    // 2. Make n_local available to all processors
    MPI_Bcast(&n_local,1,MPI_INT,0,comm);
    
    // 3. Declare local subarray that will receive scattered array_initial from manager
    int *array_local = malloc(n_local*sizeof(int));
    
    // 4. Scatter parts of the array (subarrays) to all processors
   MPI_Scatter(array,n_local,MPI_INT,array_local,n_local,MPI_INT,0,comm);
    
    // 4.1 Add a printing statement that prints, the rank, and the array_local after the scatter.
    printf("Rank = %d, Subarray :",rank);
    printArray(array_local,n_local);

    
    // 5. Each processor works on its part of the subarray ( multiply by 10)
    for(int i=0;i< n_local;i++)
    {
        array_local[i]=array_local[i] * 10;
    }

    // 6. Gather all new subarrays into manager processors 0
    MPI_Gather(array_local,n_local,MPI_INT,array_new,n_local,MPI_INT,0,comm);
    
    // 7. Manager processor prints out the final processed array
    if(rank==0)
    {
        printf("The Final output array is :");
        printArray(array_new,N);
    }
    
    MPI_Finalize();
}

/*--------------------
 Print array to screen
 */
void printArray(int a[], int N){
    printf("[");
    for (int i=0; i<N-1; i++){
        printf("%d,",a[i]);
    }
    printf("%d]\n",a[N-1]);
    
}
