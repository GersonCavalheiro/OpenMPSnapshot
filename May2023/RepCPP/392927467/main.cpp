

#include <unistd.h>
#include <omp.h>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define max_size 100000             
using namespace std;

int main(int argc, char** argv)
{


int rank;                       
int size = 100;                 
int* data;                      
int dist;                       
int nprocs;                     
int processes;                  
int* send;                      
int* recv_slave;                
int recv_master;                
int size_of_recv;               
int count = 0;                  
int search;                     
int found = 111;                
int abort = 000;                
int not_found = 666;            


MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
processes = nprocs;


if (rank == 0)                                                          
{   
MPI_Status status; 
cout<<"Enter the number you want to search = ";
cin>> search;
cout<<endl;
data = new int[size];                                               

int x = 1;
for (int i=0; i<size; i++)                                          
{
data[i] = x;
x++;
}

if ((size % (processes-1)) == 0)                                    
{
dist = size / (processes-1);
send = new int[size];   
}

for (int i =1; i< processes; i++)                                   
{
for (int j=count; j<(count+dist); j++)
{
send[j] = data[j];
}


#pragma omp critical                                                                
{
MPI_Send(send+count,dist, MPI_INT, i, 6, MPI_COMM_WORLD);                       
count+=dist;
MPI_Send(&search,1, MPI_INT, i, 6, MPI_COMM_WORLD);                             

}            
}

MPI_Recv (&recv_master, 1, MPI_INT,MPI_ANY_SOURCE,6, MPI_COMM_WORLD, &status);          

if (recv_master == found)                                                               
{
cout<<"MASTER: The value was found by SLAVE "<<status.MPI_SOURCE<<endl<<endl;

for (int i=1; i<processes; i++)
{
cout<<"MASTER: SENDING ABORT SIGNALS TO SLAVE "<<i<<endl;
MPI_Send(&abort,1, MPI_INT, i, 6, MPI_COMM_WORLD);                              
}

}
}


else
{
int search1,abort1;
bool found1 = false;
#pragma omp parallel num_threads(1)                                                     
{
recv_slave = new int[size];                                                         
MPI_Status status;                                                                  
#pragma omp critical
{
MPI_Recv (recv_slave, max_size, MPI_INT,0,6, MPI_COMM_WORLD, &status);          
MPI_Get_count(&status, MPI_INT,&size_of_recv);                                  
cout<<"SLAVE "<<rank<<": ";  

for (int i =0; i<size_of_recv; i++)
{
cout<<recv_slave[i]<<" ";                                               
}
cout<<endl<<endl<<endl;

#pragma omp_barrier                                                             
{
int idx;  
MPI_Recv (&search1, 1, MPI_INT,0,6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);     


for (int i = 0; i < size_of_recv; i++)                                      
{
if (recv_slave[i] == search1)                                           
{
idx =  i;
found1 = true;
}
}

if (found1 == true)
{
cout<<"SLAVE "<<rank <<": Value found at index "<<idx<<endl<<endl;
cout<<"SLAVE "<<rank<<": Sending FOUND signal to the master..."<<endl;
MPI_Send(&found,1, MPI_INT, 0, 6, MPI_COMM_WORLD);                     
}
else
{
sleep(2);
cout<<"SLAVE "<<rank<<": VALUE NOT FOUND!"<<endl;                      
}
}   
}
MPI_Recv (&abort1, 1, MPI_INT,0,6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);             

if (abort1 == abort)
{
cout<< "SLAVE "<<rank<<": ABORT SIGNAL RECIEVED!"<<endl;
}
}       
}
MPI_Finalize();
return 0;
}


