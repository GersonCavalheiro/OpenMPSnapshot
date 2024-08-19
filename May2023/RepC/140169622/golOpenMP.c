#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#define DIM 960
#define TESTFILE "../TestFiles/960.txt"
#define BUFFER_SIZE 1024
#define NUMBER_OF_LOOPS 150
int main(int argc, char* argv[])
{
int 	comm_sz; 		
int 	my_rank; 		
int	block_dim;
int i;
MPI_Init(NULL, NULL);
MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
MPI_Comm new_comm;
int dim_size[2],periods[2];
dim_size[0] = sqrt(comm_sz);
dim_size[1] = sqrt(comm_sz);
periods[0] = 1; 
periods[1] = 1; 
MPI_Cart_create(MPI_COMM_WORLD,2,dim_size,periods,1,&new_comm); 
block_dim = DIM/sqrt(comm_sz) + 2;
char **a, **b;  
a = malloc(block_dim*sizeof(char*));
b = malloc(block_dim*sizeof(char*));
a[0] = malloc(block_dim*block_dim*sizeof(char));
b[0] = malloc(block_dim*block_dim*sizeof(char));
for(i=1; i<block_dim; i++)
{
a[i] = &(a[0][i*block_dim]);
b[i] = &(b[0][i*block_dim]);
}
int t;
for(i = 0; i < block_dim;i++){
for(t = 0; t < block_dim;t++){
b[i][t] = '2';
} 
}
int coords[2];
MPI_Cart_coords(new_comm,my_rank,2,coords);  
MPI_Datatype mysubarray;
int array_of_sizes[2];
int array_of_subsizes[2];
int starts[2];
array_of_sizes[0] = DIM;
array_of_sizes[1] = DIM;
array_of_subsizes[0] = block_dim-2;
array_of_subsizes[1] = block_dim-2;
starts[0] = coords[0] * (block_dim-2);
starts[1] = coords[1] * (block_dim-2);
MPI_Type_create_subarray(2,array_of_sizes,array_of_subsizes,starts,MPI_ORDER_C,MPI_CHAR,&mysubarray);
MPI_Type_commit(&mysubarray);
MPI_File fh;
MPI_File_open(new_comm, TESTFILE, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
MPI_File_set_view(fh,0,MPI_CHAR,mysubarray,"native",MPI_INFO_NULL);
MPI_Request* requ;
requ = malloc((block_dim-2)*sizeof(MPI_Request));  
for(i = 1; i < block_dim - 1; i++)
MPI_File_iread(fh, &a[i][1], block_dim-2, MPI_CHAR, &requ[i-1]);
MPI_File_close(&fh);
int N,S,W,E,NW,NE,SW,SE;
int neigh_coords[2];
neigh_coords[0] = coords[0] - 1;
neigh_coords[1] = coords[1];
MPI_Cart_rank(new_comm,neigh_coords,&N);
neigh_coords[0] = coords[0] + 1;
neigh_coords[1] = coords[1];
MPI_Cart_rank(new_comm,neigh_coords,&S);
neigh_coords[0] = coords[0];
neigh_coords[1] = coords[1] + 1;
MPI_Cart_rank(new_comm,neigh_coords,&E);
neigh_coords[0] = coords[0];
neigh_coords[1] = coords[1] - 1;
MPI_Cart_rank(new_comm,neigh_coords,&W);
neigh_coords[0] = coords[0] - 1;
neigh_coords[1] = coords[1] + 1;
MPI_Cart_rank(new_comm,neigh_coords,&NE);
neigh_coords[0] = coords[0] - 1;
neigh_coords[1] = coords[1] - 1;
MPI_Cart_rank(new_comm,neigh_coords,&NW);
neigh_coords[0] = coords[0] + 1;
neigh_coords[1] = coords[1] + 1;
MPI_Cart_rank(new_comm,neigh_coords,&SE);
neigh_coords[0] = coords[0] + 1;
neigh_coords[1] = coords[1] - 1;
MPI_Cart_rank(new_comm,neigh_coords,&SW);
MPI_Datatype row,col;
MPI_Type_contiguous(block_dim-2,MPI_CHAR,&row);
MPI_Type_commit(&row);
MPI_Type_vector(block_dim-2,1,block_dim,MPI_CHAR,&col);
MPI_Type_commit(&col);
MPI_Request r[16];
MPI_Status stats[16];
MPI_Send_init(&a[1][1],1,row,N,0,new_comm,&r[0]);
MPI_Send_init(&a[block_dim-2][1],1,row,S,1,new_comm,&r[1]);
MPI_Send_init(&a[1][block_dim-2],1,col,E,2,new_comm,&r[2]);
MPI_Send_init(&a[1][1],1,col,W,3,new_comm,&r[3]);
MPI_Send_init(&a[1][block_dim-2],1,MPI_CHAR,NE,4,new_comm,&r[4]);
MPI_Send_init(&a[1][1],1,MPI_CHAR,NW,5,new_comm,&r[5]);
MPI_Send_init(&a[block_dim-2][block_dim-2],1,MPI_CHAR,SE,6,new_comm,&r[6]);
MPI_Send_init(&a[block_dim-2][1],1,MPI_CHAR,SW,7,new_comm,&r[7]);
MPI_Recv_init(&a[block_dim-1][1],1,row,S,0,new_comm,&r[8]);
MPI_Recv_init(&a[0][1],1,row,N,1,new_comm,&r[9]);
MPI_Recv_init(&a[1][0],1,col,W,2,new_comm,&r[10]);
MPI_Recv_init(&a[1][block_dim-1],1,col,E,3,new_comm,&r[11]);
MPI_Recv_init(&a[block_dim-1][0],1,MPI_CHAR,SW,4,new_comm,&r[12]);
MPI_Recv_init(&a[block_dim-1][block_dim-1],1,MPI_CHAR,SE,5,new_comm,&r[13]);
MPI_Recv_init(&a[0][0],1,MPI_CHAR,NW,6,new_comm,&r[14]);
MPI_Recv_init(&a[0][block_dim-1],1,MPI_CHAR,NE,7,new_comm,&r[15]);
MPI_Request r2[16];
MPI_Status stats2[16];
MPI_Send_init(&b[1][1],1,row,N,0,new_comm,&r2[0]);
MPI_Send_init(&b[block_dim-2][1],1,row,S,1,new_comm,&r2[1]);
MPI_Send_init(&b[1][block_dim-2],1,col,E,2,new_comm,&r2[2]);
MPI_Send_init(&b[1][1],1,col,W,3,new_comm,&r2[3]);
MPI_Send_init(&b[1][block_dim-2],1,MPI_CHAR,NE,4,new_comm,&r2[4]);
MPI_Send_init(&b[1][1],1,MPI_CHAR,NW,5,new_comm,&r2[5]);
MPI_Send_init(&b[block_dim-2][block_dim-2],1,MPI_CHAR,SE,6,new_comm,&r2[6]);
MPI_Send_init(&b[block_dim-2][1],1,MPI_CHAR,SW,7,new_comm,&r2[7]);
MPI_Recv_init(&b[block_dim-1][1],1,row,S,0,new_comm,&r2[8]);
MPI_Recv_init(&b[0][1],1,row,N,1,new_comm,&r2[9]);
MPI_Recv_init(&b[1][0],1,col,W,2,new_comm,&r2[10]);
MPI_Recv_init(&b[1][block_dim-1],1,col,E,3,new_comm,&r2[11]);
MPI_Recv_init(&b[block_dim-1][0],1,MPI_CHAR,SW,4,new_comm,&r2[12]);
MPI_Recv_init(&b[block_dim-1][block_dim-1],1,MPI_CHAR,SE,5,new_comm,&r2[13]);
MPI_Recv_init(&b[0][0],1,MPI_CHAR,NW,6,new_comm,&r2[14]);
MPI_Recv_init(&b[0][block_dim-1],1,MPI_CHAR,NE,7,new_comm,&r2[15]);
MPI_Waitall(block_dim-2,requ,MPI_STATUSES_IGNORE); 
free(requ);
int j,z;
int neighbors;
char **c;
double local_start,local_elapsed,elapsed;
MPI_Barrier(new_comm);
local_start = MPI_Wtime();
for(i=0; i<NUMBER_OF_LOOPS;i++){
if((i%2) == 0) MPI_Startall(16,r);
else MPI_Startall(16,r2);
#pragma omp parallel for schedule(static,1) num_threads(8) private(z,neighbors) 
for(j = 2; j<block_dim-2;j++)
{
for(z = 2; z<block_dim-2;z++)
{
neighbors = 0;
neighbors += (a[j-1][z-1]-'0') + (a[j-1][z]-'0') + (a[j-1][z+1]-'0') + (a[j][z+1]-'0') + (a[j+1][z+1]-'0') + (a[j+1][z]-'0') + (a[j+1][z-1]-'0') + (a[j][z-1]-'0');
if((a[j][z] == '0') && (neighbors == 3))
{
b[j][z] = '1';
}
else if(a[j][z] == '1')
{
if(neighbors < 2){
b[j][z] = '0';
}
else if(neighbors < 4)
{
b[j][z] = '1';
}
else{
b[j][z] = '0';
}
}
else b[j][z] = '0';
}
}
if((i%2) == 0) MPI_Waitall(16,r,stats);
else MPI_Waitall(16,r2,stats2);
#pragma omp parallel num_threads(8) private(neighbors) 
{ 
#pragma omp for schedule(static,1)
for(j = 1;j < block_dim-1;j++){
neighbors = 0;
neighbors += (a[j-1][0]-'0') + (a[j-1][1]-'0') + (a[j-1][2]-'0') + (a[j][2]-'0') + (a[j+1][2]-'0') + (a[j+1][1]-'0') + (a[j+1][0]-'0') + (a[j][0]-'0');
if((a[j][1] == '0') && (neighbors == 3)){
b[j][1] = '1';;
}
else if(a[j][1] == '1'){
if(neighbors < 2){
b[j][1] = '0';
}
else if(neighbors < 4){
b[j][1] = '1';
}
else{
b[j][1] = '0';
}
}
else b[j][1] = '0';
}
#pragma omp for schedule(static,1)
for(j = 1;j < block_dim-1;j++){
neighbors = 0;
neighbors += (a[j-1][block_dim-3]-'0') + (a[j-1][block_dim-2]-'0') + (a[j-1][block_dim-1]-'0') + (a[j][block_dim-1]-'0') +
(a[j+1][block_dim-1]-'0') + (a[j+1][block_dim-2]-'0') + (a[j+1][block_dim-3]-'0') + (a[j][block_dim-3]-'0');
if((a[j][block_dim-2] == '0') && (neighbors == 3)){
b[j][block_dim-2] = '1';
}
else if(a[j][block_dim-2] == '1'){
if(neighbors < 2){
b[j][block_dim-2] = '0';
}
else if(neighbors < 4){
b[j][block_dim-2] = '1';
}
else{ 
b[j][block_dim-2] = '0';  
}
}
else b[j][block_dim-2] = '0';
}
#pragma omp for schedule(static,1)
for(j = 2;j < block_dim-2;j++){
neighbors = 0;
neighbors += (a[0][j-1]-'0') + (a[1][j-1]-'0') + (a[2][j-1]-'0') + (a[2][j]-'0') + (a[2][j+1]-'0') + (a[1][j+1]-'0') + (a[0][j+1]-'0') + (a[0][j]-'0');
if((a[1][j] == '0') && (neighbors == 3)){
b[1][j] = '1';
}
else if(a[1][j] == '1'){
if(neighbors < 2){
b[1][j] = '0';
}
else if(neighbors < 4){
b[1][j] = '1';
}
else{
b[1][j] = '0';
}
}
else b[1][j] = '0';
}
#pragma omp for schedule(static,1)
for(j = 2;j < block_dim-2;j++){
neighbors = 0;
neighbors += (a[block_dim-3][j-1]-'0') + (a[block_dim-2][j-1]-'0') + (a[block_dim-1][j-1]-'0') + (a[block_dim-1][j]-'0') + 
(a[block_dim-1][j+1]-'0') + (a[block_dim-2][j+1]-'0') + (a[block_dim-3][j+1]-'0') + (a[block_dim-3][j]-'0');
if((a[block_dim-2][j] == '0') && (neighbors == 3)){
b[block_dim-2][j] = '1';
}
else if(a[block_dim-2][j] == '1'){
if(neighbors < 2){
b[block_dim-2][j] = '0';
}
else if(neighbors < 4){
b[block_dim-2][j] = '1';
}
else{
b[block_dim-2][j] = '0';
}
}
else b[block_dim-2][j] = '0';
}
}
c = a;
a = b;
b = c;
}
local_elapsed = MPI_Wtime() - local_start;
MPI_Reduce(&local_elapsed,&elapsed,1,MPI_DOUBLE,MPI_MAX,0,new_comm);
if(i < NUMBER_OF_LOOPS) i++; 
if (my_rank == 0)
{
printf( "Elapsed time: %f seconds\n",elapsed);
}
MPI_Finalize();
free(a[0]);
free(b[0]);
free(a);
free(b);
return 0;
}
