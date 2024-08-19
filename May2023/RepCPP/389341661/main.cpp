#include <mpi.h>
#include<omp.h>
#include <stdio.h>
#include <cstdlib>
#include<iostream>
#include<unistd.h>
using namespace std;

#define num_size 55	

int main(int argc, char *argv[]){

int search_num = 48;	
int* data_set1;		
int myrank, nprocs;
int dest = 1;
int master_tag = 1;	
int worker_tag = 2;	
int ind = 0;
int abort = 0;		
int signal = 0;		
int not_found = 1;	

MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);	

MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
int div = num_size/(nprocs-1);		
MPI_Status status;
if(myrank == 0) {
cout<<"Name: Areesha Tahir "<<"Roll no: 18I-1655"<<" Sec: A"<<endl;
data_set1 = new int[num_size];		
int num = 1;
for(int i = 0; i < num_size; i++){	
data_set1[i] = num;
num++;
}
cout<<"Process 0 has data input ";
for(int i = 0; i < num_size; i++){	
cout<<data_set1[i]<<" ";
}
cout<<endl;
cout<<"Master Process: Number to search "<<search_num<<endl;

while (dest < nprocs){			
#pragma omp critical 
{
MPI_Send(&search_num, 1, MPI_INT, dest, master_tag, MPI_COMM_WORLD);	
MPI_Send(&data_set1[ind], div, MPI_INT, dest, master_tag, MPI_COMM_WORLD);	
dest += 1;
ind = ind + div;						
}
}

dest = 1;
while(signal == 0){
while(dest < nprocs){
#pragma omp critical 
{
MPI_Send(&abort, 1, MPI_INT, dest, master_tag, MPI_COMM_WORLD);	
dest = dest + 1;
}
}
dest = 1;
MPI_Recv(&signal, 1, MPI_INT, MPI_ANY_SOURCE, worker_tag, MPI_COMM_WORLD, &status);
#pragma omp critical 
{
if(signal == 1){
cout<<"Master Process: Process "<<status.MPI_SOURCE<<" has found the number"<<endl;
cout<<"Informing all processes to abort"<<endl;
dest = 1;
abort = 1;
while(dest < nprocs){
MPI_Send(&abort, 1, MPI_INT, dest, master_tag, MPI_COMM_WORLD);	
dest = dest + 1;
}
sleep(2);
exit(0);
}
}
if(signal == 2){
not_found += 1; 
if(not_found < nprocs){	
signal = 0;
}
}
}
if (not_found == nprocs){
cout<<"Master Process: Number not in data set"<<endl;
exit(0);
}

}


else {
int recv_data[div];	
int sig = 0;		
int id = 0;		
int abort_sig = 0;	
bool found = true;
#pragma omp critical
{
MPI_Recv(&search_num, 1, MPI_INT, 0, master_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);	
MPI_Recv(&recv_data[id], div, MPI_INT, 0, master_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);	
sleep(1);
cout<<"Process "<<myrank<<" has data input ";	
for(int i = 0; i < (div); i++){
cout<<recv_data[i]<<" ";
}
cout<<endl;
sleep(2);
#pragma omp barrier
{
int i = 0;
for(; i<(div) ;i++){	
MPI_Recv(&abort_sig, 1, MPI_INT, 0, master_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
if (!abort_sig){	
if(recv_data[i] == search_num){		
sig = 1;		
MPI_Send(&sig, 1, MPI_INT, 0, worker_tag, MPI_COMM_WORLD);	
sleep(2);

}
if(i == (div-1)){	
found = false;
}
if(!found){	
sig = 2;	
MPI_Send(&sig, 1, MPI_INT, 0, worker_tag, MPI_COMM_WORLD);	
break;

}
if(i != (div-1) & recv_data[i] != search_num){	
MPI_Send(&sig, 1, MPI_INT, 0, worker_tag, MPI_COMM_WORLD);
}

}
}
if(sig != 0) {	
while(!abort_sig){
MPI_Recv(&abort_sig, 1, MPI_INT, 0, master_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
if(abort_sig){
sleep(1);
cout<<"PROCESS "<<myrank<<" SEARCH ABORTED"<<endl;
break;
}
}
}

}
}	
}

MPI_Finalize();
return 0;
}
