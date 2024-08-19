#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include "utils.h"
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define isroot if(rank==0)
void SpGEMM_bigslice(int  *Acol, int *Arow, int An, 
int  *Bcol, int *Brow, int Bm,
int **Ccol, int *Crow, int *Csize,
int start_row, int end_row)
{
int nnzcum=0;                       
bool *xb = calloc(Bm,sizeof(bool)); 
int ip=0;                           
for(int i=start_row; i<end_row; i++){
int nnzpv = nnzcum;             
Crow[ip++] = nnzcum;            
if(nnzcum + Bm > *Csize){       
*Csize += MAX(Bm, *Csize/4);
*Ccol = realloc(*Ccol,*Csize*sizeof(int));
}
for(int jj=Arow[i]; jj<Arow[i+1]; jj++){    
int j = Acol[jj];
for(int kp=Brow[j]; kp<Brow[j+1]; kp++){
int k = Bcol[kp];
if(!xb[k]){
xb[k] = true;
(*Ccol)[nnzcum] = k;
nnzcum++;
}
}
}
if(nnzcum > nnzpv){                         
quickSort(*Ccol,nnzpv,nnzcum-1);            
for(int p=nnzpv; p<nnzcum; p++){        
xb[ (*Ccol)[p] ] = false;
}
}
}
Crow[ip] = nnzcum;
free(xb);
}
void SpGEMM_omp(int  *Acol, int *Arow, int An, 
int  *Bcol, int *Brow, int Bm,
int **Ccol, int *Crow,
int tBlock)
{
int slices = An/tBlock; 
int  *Ccol_sizes  = malloc(slices*sizeof(int ));    
int **Ccol_tBlock = malloc(slices*sizeof(int*));    
int **Crow_tBlock = malloc(slices*sizeof(int*));    
uint32_t i, init_size = Bm; 
for(i=0; i<slices; i++){
Ccol_sizes[i]  = init_size;
Ccol_tBlock[i] = malloc(init_size*sizeof(int));
Crow_tBlock[i] = malloc((tBlock+1)*sizeof(int));
}
#pragma omp parallel private(i)
{
#pragma omp for schedule(static) nowait
for( i=0; i<slices; i++ ){
SpGEMM_bigslice(Acol,Arow,An,
Bcol,Brow,Bm,
&Ccol_tBlock[i],Crow_tBlock[i],&Ccol_sizes[i],
i*tBlock, (i+1)*tBlock);
}
}
int nnz =0;
for(i=0; i<slices; i++){
nnz += Crow_tBlock[i][tBlock];
}
*Ccol = malloc(nnz*sizeof(int));
Crow[0] = 0;
int nnzcum = 0; 
for(i=0; i<slices; i++){
memcpy(&(*Ccol)[nnzcum], Ccol_tBlock[i], Crow_tBlock[i][tBlock]*sizeof(int));
memcpy(&Crow[i*tBlock+1], &Crow_tBlock[i][1], tBlock*sizeof(int));
nnzcum += Crow_tBlock[i][tBlock];
free(Ccol_tBlock[i]);
free(Crow_tBlock[i]);
}
free(Ccol_sizes);
free(Ccol_tBlock);
free(Crow_tBlock);
int row_sum = Crow[tBlock];
for(i=1; i<slices; i++){
for(int j=1; j<tBlock+1; j++){
Crow[i*tBlock + j] += row_sum;
}
row_sum = Crow[(i+1)*tBlock];
}    
}
void SpGEMM_mpi(int  *Acol, int *Arow, int An, 
int  *Bcol, int *Brow, int Bm,
int **Ccol, int *Crow,
int tBlock)
{
int numtasks, rank;
MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
int tasksize = An/numtasks; 
int *Ccol_slice;    
int *Crow_slice = malloc((tasksize+1)*sizeof(int));
SpGEMM_omp(Acol,&Arow[rank*tasksize],tasksize,
Bcol,Brow,Bm,
&Ccol_slice,Crow_slice,
tBlock);
int all_nnz=0;
MPI_Reduce(&Crow_slice[tasksize],&all_nnz,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
int *reccounts;
isroot{
reccounts = malloc(numtasks*sizeof(int));
}
MPI_Gather(&Crow_slice[tasksize],1,MPI_INT,reccounts,1,MPI_INT,0,MPI_COMM_WORLD);
int *disps;
isroot{
disps = malloc(numtasks*sizeof(int));
disps[0] = 0;
for(int t=1; t<numtasks; t++){
disps[t] = disps[t-1] + reccounts[t-1];
}
}
isroot{
*Ccol = malloc(all_nnz*sizeof(int));
Crow[0] = 0;
}
MPI_Gatherv(Ccol_slice,Crow_slice[tasksize],MPI_INT,*Ccol,reccounts,disps,MPI_INT,0,MPI_COMM_WORLD);
MPI_Gather(&Crow_slice[1],tasksize,MPI_INT,&Crow[1],tasksize,MPI_INT,0,MPI_COMM_WORLD);             
free(Crow_slice);
free(Ccol_slice);
isroot{
free(disps);
free(reccounts);
int row_sum = Crow[tasksize];
for(int i=1; i<numtasks; i++){
for(int j=1; j<tasksize+1; j++){
Crow[i*tasksize + j] += row_sum;
}
row_sum = Crow[(i+1)*tasksize];
}
}
}
void SpGEMM_masked(int  *Acol, int *Arow, int An, 
int  *Bcol, int *Brow, int Bm,
int  *Fcol, int *Frow,
int **Ccol, int *Crow, int *Csize)
{
int nnzcum=0;
bool *xb = malloc(An*sizeof(bool));
for(int i=0; i<An; i++) xb[i] = true;           
for(int i=0; i<An; i++){
int nnzpv = nnzcum; 
Crow[i] = nnzcum;
if(nnzcum + An > *Csize){
*Csize += MAX(An, *Csize/4);
*Ccol = realloc(*Ccol,*Csize*sizeof(int));
}
for(int jj=Frow[i]; jj<Frow[i+1]; jj++){    
xb[ Fcol[jj] ] = false;                 
}                                           
for(int jj=Arow[i]; jj<Arow[i+1]; jj++){
int j = Acol[jj];
for(int kp=Brow[j]; kp<Brow[j+1]; kp++){
int k = Bcol[kp];
if(!xb[k]){
xb[k] = true;
(*Ccol)[nnzcum] = k;
nnzcum++;
}
}
}
if(nnzcum > nnzpv){
quickSort(*Ccol,nnzpv,nnzcum-1);
for(int p=nnzpv; p<nnzcum; p++){
xb[ (*Ccol)[p] ] = false;
}
}
for(int jj=Frow[i]; jj<Frow[i+1]; jj++){   
xb[ Fcol[jj] ] = true;             
}
}
Crow[An] = nnzcum;
free(xb);
}
bool SpGEMM_valid(int *Acol, int *Arow, int *Bcol, int *Brow, int n){
for(int i=0; i<=n; i++){
if(Arow[i]!=Brow[i]){
return false;
}
}
for(int i=0; i<Arow[n]; i++){
if(Acol[i]!=Bcol[i]){
return false;
}
}
return true;
}
int test_mpi(int argc, char const *argv[]){
int numtasks, rank;
MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
int tBlock  = atoi(argv[2]);
int threads = atoi(argv[3]);
omp_set_num_threads(threads);
uint32_t *Arow,*Acol;
int An,Am,Annz;
readCOO(argv[1],&Arow,&Acol,&An,&Am,&Annz);
int *nCrow = calloc((An+1),sizeof(int));
int *nCcol;
MPI_Barrier(MPI_COMM_WORLD);    
SpGEMM_mpi(Acol,Arow,An,Acol,Arow,An,&nCcol,nCrow,tBlock);
isroot{
int tCsize = nCrow[An];
int *tCrow = calloc((An+1),sizeof(int));
int *tCcol = malloc(tCsize*sizeof(int));
SpGEMM_bigslice(Acol,Arow,An,Acol,Arow,An,&tCcol,tCrow,&tCsize,0,An);
if(SpGEMM_valid(nCcol,nCrow,tCcol,tCrow,An)){
printf("Results of serial and multricore are the same!\n");
}else{
printf("The results dont match\n");
}
}
free(Acol);
free(Arow);
isroot free(nCcol);
free(nCrow);
}
int main(int argc, char const *argv[])
{
int numtasks, rank, provided;
MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
MPI_Query_thread(&provided);
MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
if(argc!=4){
printf("usage: mpirun  -n  numtasks  SpGEMM_mpi_omp  path-to-matrix  threadslice_size  number_of_threads\n");
exit(1);
}
test_mpi(argc,argv);
MPI_Finalize();
return 0;
}
