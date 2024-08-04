/*
File Name : mpi_sum.c

Name : Yasoda Krishna Reddy Annapureddy
Course : Introducation to Parallel Processing
HomeWork 2 MPI SUM

*/
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char* argv[]){
    // This conditional statement checks the user had entered N value or not
    if(argc!=2){
        printf("The arguments are incorrect \n");
        exit(1);
    }
    
    //Declaring the variables
    int my_rank, nproc;
    int N=atoi(argv[1]);
    int start,end;
    int sum,local_sum;
    sum=0;
    
    
    // Creating the basic MPI init, comm, status, size and rank variables
    MPI_Init(&argc,&argv);
    MPI_Comm comm; MPI_Status status; comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &my_rank);

    //Calculating the starting value for each process
    start=N*my_rank/nproc+1;
    //Calculating the end value for each process
    end=N*(my_rank+1)/nproc;
    //This for loop is for calculating the sum of numbers from start to end in each process
    for(int i=start;i<=end;++i){
        sum=sum+i;
    }
    //print statement prints the sum at each process
    printf("Rank : %d --> Partial Sum : %d\n",my_rank,sum);
    //This conditional statement checks the process is manager or not
    if(my_rank==0){
	//This for loop will receive all the local_sum of other process and add them to sum variable in manager process.
        for(int j=1;j<nproc;j++){
            MPI_Recv(&local_sum,1,MPI_INT,j,1,MPI_COMM_WORLD,&status);
            sum=sum+local_sum;
        }
    }
    else{
	//This send statement will send all the local the local sum of each process to manager process
        MPI_Send(&sum,1,MPI_INT,0,1,MPI_COMM_WORLD);
    }
    //Calculate the sum using serial process and prints the serial sum
    if(my_rank==0){
        printf("Total Sum using Parallel Approach = %d\n",sum);
        int serial_sum=0;
        for(int i=1;i<=N;i++){
            serial_sum=serial_sum+i;
        }
        printf("Total Sum using Serial Approach = %d\n",serial_sum);
    }
    
    MPI_Finalize();
}
