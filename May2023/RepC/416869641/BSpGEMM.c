#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include "utils.h"
#include "ins.h"
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
bool SpGEMM(int  *Acol, int *Arow, int An, 
int  *Bcol, int *Brow, int Bm,
int **Ccol, int *Crow, int *Csize)
{
for(int i=0; i<An; i++){
for(int jj=Arow[i]; jj<Arow[i+1]; jj++){
int j = Acol[jj];
for(int k=Brow[j]; k<Brow[j+1]; k++){
insertCSR_mv(Ccol,Crow,Bcol[k],Csize,i,An); 
}
}
}
return Crow[An];
}
bool SpGEMM_d(int  *Acol, int *Arow, int An, 
int  *Bcol, int *Brow, int Bm,
int **Ccol, int *Crow, int *Csize)
{
int nnzcum=0;
bool *xb = calloc(An,sizeof(bool));
for(int i=0; i<An; i++){
int nnzpv = nnzcum;
Crow[i] = nnzcum;
if(nnzcum + An > *Csize){
*Csize += MAX(An, *Csize/4);
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
Crow[An] = nnzcum;
free(xb);
return Crow[An];
}
bool SpGEMM_dor(int  *Acol, int *Arow, int An, 
int  *Bcol, int *Brow, int Bm,
int **Ccol, int *Crow, int *Csize,
int  *Dcol, int *Drow)
{
int nnzcum=0;
bool *xb = calloc(An,sizeof(bool));
for(int i=0; i<An; i++){
int nnzpv = nnzcum;
Crow[i] = nnzcum;
if(nnzcum + An > *Csize){
*Csize += MAX(An, *Csize/4);
*Ccol = realloc(*Ccol,*Csize*sizeof(int));
}
for(int jj=Drow[i]; jj<Drow[i+1]; jj++){ 
int j = Dcol[jj];
xb[j] = true;
(*Ccol)[nnzcum] = j;
nnzcum++;
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
Crow[An] = nnzcum;
free(xb);
return Crow[An];
}
bool SpGEMM_dor_cpy(int  *Acol, int *Arow, int An, 
int  *Bcol, int *Brow, int Bm,
int **Ccol, int *Crow, int *Csize,
int  *Dcol, int *Drow)
{
int nnzcum=0;
bool *xb = calloc(An,sizeof(bool));
for(int i=0; i<An; i++){
int nnzpv = nnzcum;
Crow[i] = nnzcum;
if(nnzcum + An > *Csize){
*Csize += MAX(An, *Csize/4);
*Ccol = realloc(*Ccol,*Csize*sizeof(int));
}
for(int jj=Drow[i]; jj<Drow[i+1]; jj++){ 
int j = Dcol[jj];
xb[j] = true;
}
const int to_move = Drow[i+1]-Drow[i];
memcpy(&((*Ccol)[nnzcum]),&Dcol[Drow[i]],to_move*sizeof(int));
nnzcum += to_move;
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
Crow[An] = nnzcum;
free(xb);
return Crow[An];
}
bool SpGEMM_dor_masked( int  *Acol, int *Arow, int An, 
int  *Bcol, int *Brow, int Bm,
int  *Fcol, int *Frow,
int **Ccol, int *Crow, int *Csize,
int  *Dcol, int *Drow)
{
int nnzcum=0;
bool *xb = calloc(An,sizeof(bool));
bool *mb = calloc(Bm,sizeof(bool));
for(int i=0; i<An; i++){
int nnzpv = nnzcum;
Crow[i] = nnzcum;
if(nnzcum + An > *Csize){
*Csize += MAX(An, *Csize/4);
*Ccol = realloc(*Ccol,*Csize*sizeof(int));
}
for(int jj=Frow[i]; jj<Frow[i+1]; jj++){ 
int j = Fcol[jj];
mb[j] = true;
}
for(int jj=Drow[i]; jj<Drow[i+1]; jj++){ 
int j = Dcol[jj];
if(mb[j]){
xb[j] = true;
(*Ccol)[nnzcum] = j;
nnzcum++;
}
}
for(int jj=Arow[i]; jj<Arow[i+1]; jj++){
int j = Acol[jj];
for(int kp=Brow[j]; kp<Brow[j+1]; kp++){
int k = Bcol[kp];
if(mb[k] && !xb[k]){
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
int j = Fcol[jj];
mb[j] = false;
}
}
Crow[An] = nnzcum;
free(xb);
free(mb);
return nnzcum;
}
bool SpGEMM_d_masked( int  *Acol, int *Arow, int An, 
int  *Bcol, int *Brow, int Bm,
int  *Fcol, int *Frow,
int **Ccol, int *Crow, int *Csize)
{
int nnzcum=0;
bool *xb = calloc(An,sizeof(bool));
for(int i=0; i<An; i++){
int nnzpv = nnzcum;
Crow[i] = nnzcum;
if(nnzcum + An > *Csize){
*Csize += MAX(An, *Csize/4);
*Ccol = realloc(*Ccol,*Csize*sizeof(int));
}
for(int jj=Frow[i]; jj<Frow[i+1]; jj++){
xb[ Fcol[jj] ] = true;
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
xb[ Fcol[jj] ] = false;
}
}
Crow[An] = nnzcum;
free(xb);
return nnzcum;
}
bool SpGEMM_dor_nosort( int  *Acol, int *Arow, int An, 
int  *Bcol, int *Brow, int Bm,
int **Ccol, int *Crow, int *Csize,
int  *Dcol, int *Drow)
{
int nnzcum=0;
bool *xb = calloc(An,sizeof(bool));
for(int i=0; i<An; i++){
const int nnzpv = nnzcum;
Crow[i] = nnzcum;
if(nnzcum + An > *Csize){
*Csize += MAX(An, *Csize/4);
*Ccol = realloc(*Ccol,*Csize*sizeof(int));
}
for(int jj=Drow[i]; jj<Drow[i+1]; jj++){ 
int j = Dcol[jj];
xb[j] = true;
(*Ccol)[nnzcum] = j;
nnzcum++;
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
for(int p=nnzpv; p<nnzcum; p++){
xb[ (*Ccol)[p] ] = false;
}
}
}
Crow[An] = nnzcum;
free(xb);
return Crow[An];
}
void BSpGEMM(CSRbCSR *A,CSRbCSR *B, CSRbCSR *C, int b){
struct timespec mS,mE,rbS,rbE;
int bpc = A->n / b;
int bpr = B->m / b;
int NB  = bpc*bpr; 
int init_size = A->nnz;
int  *Ccol_size = malloc(NB*sizeof(int));
int **Ccol_dist = malloc(NB*sizeof(int*));
int **Crow_dist = malloc(NB*sizeof(int*));
bool is_nzb;
bool *first_time = malloc(NB*sizeof(bool));
for(int i=0; i<NB; i++) first_time[i]=1;
C->n = A->n;
C->bpc = bpc;
C->bpr = bpr;
int CBsize = A->nnz;
C->Bcol = malloc(CBsize*sizeof(int));
C->Brow = calloc((C->bpc+1),sizeof(int));
clock_gettime(CLOCK_MONOTONIC, &mS);
for(int i=0; i<bpc; i++){
for(int jj=A->Brow[i]; jj<A->Brow[i+1]; jj++){
int j = A->Bcol[jj];
for(int k=B->Brow[j]; k<B->Brow[j+1]; k++){
int Aptr = b*jj;
int Bptr = b*k;
int blk = i*bpr + B->Bcol[k];
if(first_time[blk]){
first_time[blk]=0;
Ccol_size[blk] = init_size;
Ccol_dist[blk] = malloc(init_size*sizeof(int));
Crow_dist[blk] = calloc(b+1, sizeof(int));
}
is_nzb = SpGEMM(A->LLcol,&A->LLrow[Aptr],b,B->LLcol,&B->LLrow[Bptr],b,&Ccol_dist[blk],Crow_dist[blk],&Ccol_size[blk]);
if(is_nzb) insertCSR_mv(&C->Bcol,C->Brow,B->Bcol[k],&CBsize,i,C->bpc);
}
}
}
clock_gettime(CLOCK_MONOTONIC, &mE);
printf("multiply: %lf\n",timeConv(diff(mS,mE)));
clock_gettime(CLOCK_MONOTONIC, &rbS);
C->nnz=0;
for(int i=0; i<bpc; i++){
for(int j=C->Brow[i]; j<C->Brow[i+1]; j++){
int blk = i*bpr + C->Bcol[j];
C->nnz += Crow_dist[blk][b];
}
}
C->LLcol = malloc(C->nnz*sizeof(int));
C->LLrow = malloc((C->n+1)*sizeof(int));
C->LLrow[0] = 0;
int nzb_cnt = 0;
for(int iB=0; iB<bpc; iB++){
for(int i=0; i<b; i++){
for(int jB=C->Brow[iB]; jB<C->Brow[iB+1]; jB++){
int bcol = C->Bcol[jB];
int blk = iB*bpr + bcol;
for(int j=Crow_dist[blk][i]; j<Crow_dist[blk][i+1]; j++){
C->LLcol[nzb_cnt] = bcol*b + Ccol_dist[blk][j];
nzb_cnt++;
}
if( i==(b-1) ){
free(Crow_dist[blk]);
free(Ccol_dist[blk]);
}    
}
C->LLrow[iB*b + i + 1] = nzb_cnt;
}
}
clock_gettime(CLOCK_MONOTONIC, &rbE);
printf("rebuild: %lf\n",timeConv(diff(rbS,rbE)));
printf("nnz=%d\n",C->LLrow[C->n]);
}
void BSpGEMM_d(CSRbCSR *A,CSRbCSR *B, CSRbCSR *C, int b){
struct timespec mS,mE,rbS,rbE;
int bpc = A->n / b;
int bpr = B->m / b;
int NB  = bpc*bpr; 
int init_size = A->nnz;
int  *Ccol_size1 = malloc(NB*sizeof(int)); 
int  *Ccol_size2 = malloc(NB*sizeof(int));
int  *Ccol_size3 = malloc(NB*sizeof(int));
int **Ccol_dist1 = malloc(NB*sizeof(int*));
int **Ccol_dist2 = malloc(NB*sizeof(int*));
int **Ccol_dist3 = malloc(NB*sizeof(int*));
int **Crow_dist1 = malloc(NB*sizeof(int*));
int **Crow_dist2 = malloc(NB*sizeof(int*));
int **Crow_dist3 = malloc(NB*sizeof(int*));
bool is_nzb;
bool *first_time = malloc(NB*sizeof(bool));
bool *cnt        = calloc(NB,sizeof(bool));
for(int i=0; i<NB; i++) first_time[i]=1;
C->n = A->n;
C->bpc = bpc;
C->bpr = bpr;
int CBsize = A->nnz;
C->Bcol = malloc(CBsize*sizeof(int));
C->Brow = calloc((C->bpc+1),sizeof(int));
int Vsize = A->nnzb;
int *Vcol = malloc(Vsize*sizeof(int));
int *Vrow = calloc((bpc+1),sizeof(int));
clock_gettime(CLOCK_MONOTONIC, &mS);
for(int i=0; i<bpc; i++){
for(int jj=A->Brow[i]; jj<A->Brow[i+1]; jj++){
int j = A->Bcol[jj];
for(int k=B->Brow[j]; k<B->Brow[j+1]; k++){
int Aptr = b*jj;
int Bptr = b*k;
int blk = i*bpr + B->Bcol[k];
if(first_time[blk]){
first_time[blk]=0;
Ccol_size1[blk] = init_size;
Ccol_size2[blk] = init_size;
Ccol_size3[blk] = init_size;
Ccol_dist1[blk] = malloc(init_size*sizeof(int));
Ccol_dist2[blk] = malloc(init_size*sizeof(int));
Ccol_dist3[blk] = malloc(init_size*sizeof(int));
Crow_dist1[blk] = calloc((b+1),sizeof(int));
Crow_dist2[blk] = calloc((b+1),sizeof(int));
Crow_dist3[blk] = calloc((b+1),sizeof(int));
insertCSR_mv(&Vcol,Vrow,B->Bcol[k],&Vsize,i,bpc);
}
is_nzb = SpGEMM_d(A->LLcol,&A->LLrow[Aptr],b,B->LLcol,&B->LLrow[Bptr],b,&Ccol_dist1[blk],Crow_dist1[blk],&Ccol_size1[blk]);
if(is_nzb) insertCSR_mv(&C->Bcol,C->Brow,B->Bcol[k],&CBsize,i,C->bpc);
if(!cnt[blk]){ 
SpM_OR( Ccol_dist1[blk], Crow_dist1[blk], b,
Ccol_dist2[blk], Crow_dist2[blk],
&Ccol_dist3[blk], Crow_dist3[blk], &Ccol_size3[blk]);
cnt[blk]=1;
}else{
SpM_OR( Ccol_dist1[blk], Crow_dist1[blk], b,
Ccol_dist3[blk], Crow_dist3[blk],
&Ccol_dist2[blk], Crow_dist2[blk], &Ccol_size2[blk]);
cnt[blk]=0;
}
}
}
for(int jB=Vrow[i]; jB<Vrow[i+1]; jB++){ 
int blk= i*bpr + Vcol[jB];
if(!first_time[blk]){
int *temp_col_ptr;
int *temp_row_ptr;
if(!cnt[blk]){
temp_col_ptr = Ccol_dist1[blk];
Ccol_dist1[blk] = Ccol_dist2[blk];
Ccol_dist2[blk] = temp_col_ptr;
temp_row_ptr = Crow_dist1[blk];
Crow_dist1[blk] = Crow_dist2[blk];
Crow_dist2[blk] = temp_row_ptr;
}else{
temp_col_ptr = Ccol_dist1[blk];
Ccol_dist1[blk] = Ccol_dist3[blk];
Ccol_dist3[blk] = temp_col_ptr;
temp_row_ptr = Crow_dist1[blk];
Crow_dist1[blk] = Crow_dist3[blk];
Crow_dist3[blk] = temp_row_ptr;
}
free(Ccol_dist2[blk]);
free(Ccol_dist3[blk]);
free(Crow_dist2[blk]);
free(Crow_dist3[blk]);
}
}
}
clock_gettime(CLOCK_MONOTONIC, &mE);
printf("multiply: %lf\n",timeConv(diff(mS,mE)));
clock_gettime(CLOCK_MONOTONIC, &rbS);
C->nnz=0;
for(int i=0; i<bpc; i++){
for(int j=C->Brow[i]; j<C->Brow[i+1]; j++){
int blk = i*bpr + C->Bcol[j];
C->nnz += Crow_dist1[blk][b];
}
}
C->LLcol = malloc(C->nnz*sizeof(int));
C->LLrow = malloc((C->n+1)*sizeof(int));
C->LLrow[0] = 0;
int nzb_cnt = 0;
for(int iB=0; iB<bpc; iB++){
for(int i=0; i<b; i++){
for(int jB=C->Brow[iB]; jB<C->Brow[iB+1]; jB++){
int bcol = C->Bcol[jB];
int blk = iB*bpr + bcol;
for(int j=Crow_dist1[blk][i]; j<Crow_dist1[blk][i+1]; j++){
C->LLcol[nzb_cnt] = bcol*b + Ccol_dist1[blk][j];
nzb_cnt++;
}
if( i==(b-1) ){
free(Crow_dist1[blk]);
free(Ccol_dist1[blk]);
}    
}
C->LLrow[iB*b + i + 1] = nzb_cnt;
}
}
clock_gettime(CLOCK_MONOTONIC, &rbE);
printf("rebuild: %lf\n",timeConv(diff(rbS,rbE)));
printf("nnz=%d\n",C->LLrow[C->n]);
}
void BSpGEMM_d_masked(CSRbCSR *A,CSRbCSR *B, CSRbCSR *C, int b){
struct timespec mS,mE,rbS,rbE;
int bpc = A->n / b;
int bpr = B->m / b;
int NB  = bpc*bpr; 
int init_size = A->nnz;
int  *Ccol_size1 = malloc(NB*sizeof(int)); 
int  *Ccol_size2 = malloc(NB*sizeof(int));
int  *Ccol_size3 = malloc(NB*sizeof(int));
int **Ccol_dist1 = malloc(NB*sizeof(int*));
int **Ccol_dist2 = malloc(NB*sizeof(int*));
int **Ccol_dist3 = malloc(NB*sizeof(int*));
int **Crow_dist1 = malloc(NB*sizeof(int*));
int **Crow_dist2 = malloc(NB*sizeof(int*));
int **Crow_dist3 = malloc(NB*sizeof(int*));
bool is_nzb;
bool *first_time = malloc(NB*sizeof(bool));
bool *cnt        = calloc(NB,sizeof(bool));
for(int i=0; i<NB; i++) first_time[i]=1;
C->n = A->n;
C->bpc = bpc;
C->bpr = bpr;
int CBsize = A->nnz;
C->Bcol = malloc(CBsize*sizeof(int));
C->Brow = calloc((C->bpc+1),sizeof(int));
int Vsize = A->nnzb;
int *Vcol = malloc(Vsize*sizeof(int));
int *Vrow = calloc((bpc+1),sizeof(int));
int *temp;
clock_gettime(CLOCK_MONOTONIC, &mS);
for(int i=0; i<bpc; i++){
for(int jj=A->Brow[i]; jj<A->Brow[i+1]; jj++){
int j = A->Bcol[jj];
for(int k=B->Brow[j]; k<B->Brow[j+1]; k++){
int Aptr = b*jj;
int Bptr = b*k;
int blk = i*bpr + B->Bcol[k];
if(first_time[blk]){
first_time[blk]=0;
Ccol_size1[blk] = init_size;
Ccol_size2[blk] = init_size;
Ccol_size3[blk] = init_size;
Ccol_dist1[blk] = malloc(init_size*sizeof(int));
Ccol_dist2[blk] = malloc(init_size*sizeof(int));
Ccol_dist3[blk] = malloc(init_size*sizeof(int));
Crow_dist1[blk] = calloc((b+1),sizeof(int));
Crow_dist2[blk] = calloc((b+1),sizeof(int));
Crow_dist3[blk] = calloc((b+1),sizeof(int));
insertCSR_mv(&Vcol,Vrow,B->Bcol[k],&Vsize,i,bpc);
}
is_nzb = SpGEMM_d_masked(A->LLcol,&A->LLrow[Aptr],b,
B->LLcol,&B->LLrow[Bptr],b,
Ccol_dist2[blk],Crow_dist2[blk],
&Ccol_dist1[blk],Crow_dist1[blk],&Ccol_size1[blk]);
if(is_nzb) insertCSR_mv(&C->Bcol,C->Brow,B->Bcol[k],&CBsize,i,C->bpc);
SpM_OR( Ccol_dist1[blk], Crow_dist1[blk], b,
Ccol_dist2[blk], Crow_dist2[blk],
&Ccol_dist3[blk], Crow_dist3[blk], &Ccol_size3[blk]);
temp = Ccol_dist2[blk]; 
Ccol_dist2[blk] = Ccol_dist3[blk];
Ccol_dist3[blk] = temp;
temp = Crow_dist2[blk];
Crow_dist2[blk] = Crow_dist3[blk];
Crow_dist3[blk] = temp;
swap(&Ccol_size3[blk],&Ccol_size2[blk]); 
}
}
for(int jB=Vrow[i]; jB<Vrow[i+1]; jB++){ 
int blk= i*bpr + Vcol[jB];
if(!first_time[blk]){
free(Ccol_dist1[blk]);
free(Ccol_dist3[blk]);
free(Crow_dist1[blk]);
free(Crow_dist3[blk]);
}
}
}
clock_gettime(CLOCK_MONOTONIC, &mE);
printf("multiply: %lf\n",timeConv(diff(mS,mE)));
clock_gettime(CLOCK_MONOTONIC, &rbS);
C->nnz=0;
for(int i=0; i<bpc; i++){
for(int j=C->Brow[i]; j<C->Brow[i+1]; j++){
int blk = i*bpr + C->Bcol[j];
C->nnz += Crow_dist2[blk][b];
}
}
C->LLcol = malloc(C->nnz*sizeof(int));
C->LLrow = malloc((C->n+1)*sizeof(int));
C->LLrow[0] = 0;
int nzb_cnt = 0;
for(int iB=0; iB<bpc; iB++){
for(int i=0; i<b; i++){
for(int jB=C->Brow[iB]; jB<C->Brow[iB+1]; jB++){
int bcol = C->Bcol[jB];
int blk = iB*bpr + bcol;
for(int j=Crow_dist2[blk][i]; j<Crow_dist2[blk][i+1]; j++){
C->LLcol[nzb_cnt] = bcol*b + Ccol_dist2[blk][j];
nzb_cnt++;
}
if( i==(b-1) ){
free(Crow_dist2[blk]);
free(Ccol_dist2[blk]);
}    
}
C->LLrow[iB*b + i + 1] = nzb_cnt;
}
}
clock_gettime(CLOCK_MONOTONIC, &rbE);
printf("rebuild: %lf\n",timeConv(diff(rbS,rbE)));
printf("nnz=%d\n",C->LLrow[C->n]);
}
void BSpGEMM_dd(CSRbCSR *A,CSRbCSR *B, CSRbCSR *C, int b){
struct timespec mS,mE,rbS,rbE;
int bpc = A->n / b;
int bpr = B->m / b;
int NB  = bpc*bpr; 
int init_size = A->nnz;
int  *Ccol_size1 = malloc(NB*sizeof(int)); 
int  *Ccol_size2 = malloc(NB*sizeof(int));
int  *Ccol_size3 = malloc(NB*sizeof(int));
int **Ccol_dist1 = malloc(NB*sizeof(int*));
int **Ccol_dist2 = malloc(NB*sizeof(int*));
int **Ccol_dist3 = malloc(NB*sizeof(int*));
int **Crow_dist1 = malloc(NB*sizeof(int*));
int **Crow_dist2 = malloc(NB*sizeof(int*));
int **Crow_dist3 = malloc(NB*sizeof(int*));
bool is_nzb;
bool *first_time = malloc(NB*sizeof(bool));
bool *cnt        = calloc(NB,sizeof(bool));
for(int i=0; i<NB; i++) first_time[i]=1;
C->n = A->n;
C->bpc = bpc;
C->bpr = bpr;
int CBsize = A->nnz;
C->Bcol = malloc(CBsize*sizeof(int));
C->Brow = calloc((C->bpc+1),sizeof(int));
int Vsize = A->nnzb;
int *Vcol = malloc(Vsize*sizeof(int));
int *Vrow = calloc((bpc+1),sizeof(int));
int Cnnzbcum = 0;
bool *xbC = calloc(bpr,sizeof(bool));
clock_gettime(CLOCK_MONOTONIC, &mS);
for(int i=0; i<bpc; i++){
for(int jj=A->Brow[i]; jj<A->Brow[i+1]; jj++){
int j = A->Bcol[jj];
for(int k=B->Brow[j]; k<B->Brow[j+1]; k++){
int Aptr = b*jj;
int Bptr = b*k;
int blk = i*bpr + B->Bcol[k];
if(first_time[blk]){
first_time[blk]=0;
Ccol_size1[blk] = init_size;
Ccol_size2[blk] = init_size;
Ccol_size3[blk] = init_size;
Ccol_dist1[blk] = malloc(init_size*sizeof(int));
Ccol_dist2[blk] = malloc(init_size*sizeof(int));
Ccol_dist3[blk] = malloc(init_size*sizeof(int));
Crow_dist1[blk] = calloc((b+1),sizeof(int));
Crow_dist2[blk] = calloc((b+1),sizeof(int));
Crow_dist3[blk] = calloc((b+1),sizeof(int));
insertCSR_mv(&Vcol,Vrow,B->Bcol[k],&Vsize,i,bpc);
}
is_nzb = SpGEMM(A->LLcol,&A->LLrow[Aptr],b,B->LLcol,&B->LLrow[Bptr],b,&Ccol_dist1[blk],Crow_dist1[blk],&Ccol_size1[blk]);
if(is_nzb) insertCSR_mv(&C->Bcol,C->Brow,B->Bcol[k],&CBsize,i,C->bpc);
if(!cnt[blk]){ 
SpM_OR( Ccol_dist1[blk], Crow_dist1[blk], b,
Ccol_dist2[blk], Crow_dist2[blk],
&Ccol_dist3[blk], Crow_dist3[blk], &Ccol_size3[blk]);
cnt[blk]=1;
}else{
SpM_OR( Ccol_dist1[blk], Crow_dist1[blk], b,
Ccol_dist3[blk], Crow_dist3[blk],
&Ccol_dist2[blk], Crow_dist2[blk], &Ccol_size2[blk]);
cnt[blk]=0;
}
}
}
for(int jB=Vrow[i]; jB<Vrow[i+1]; jB++){ 
int blk= i*bpr + Vcol[jB];
if(!first_time[blk]){
int *temp_col_ptr;
int *temp_row_ptr;
if(cnt==0){
temp_col_ptr = Ccol_dist1[blk];
Ccol_dist1[blk] = Ccol_dist2[blk];
Ccol_dist2[blk] = temp_col_ptr;
temp_row_ptr = Crow_dist1[blk];
Crow_dist1[blk] = Crow_dist2[blk];
Crow_dist2[blk] = temp_row_ptr;
}else{
temp_col_ptr = Ccol_dist1[blk];
Ccol_dist1[blk] = Ccol_dist3[blk];
Ccol_dist3[blk] = temp_col_ptr;
temp_row_ptr = Crow_dist1[blk];
Crow_dist1[blk] = Crow_dist3[blk];
Crow_dist3[blk] = temp_row_ptr;
}
free(Ccol_dist2[blk]);
free(Ccol_dist3[blk]);
free(Crow_dist2[blk]);
free(Crow_dist3[blk]);
}
}
}
clock_gettime(CLOCK_MONOTONIC, &mE);
printf("multiply: %lf\n",timeConv(diff(mS,mE)));
clock_gettime(CLOCK_MONOTONIC, &rbS);
C->nnz=0;
for(int i=0; i<bpc; i++){
for(int j=C->Brow[i]; j<C->Brow[i+1]; j++){
int blk = i*bpr + C->Bcol[j];
C->nnz += Crow_dist1[blk][b];
}
}
C->LLcol = malloc(C->nnz*sizeof(int));
C->LLrow = malloc((C->n+1)*sizeof(int));
C->LLrow[0] = 0;
int nzb_cnt = 0;
for(int iB=0; iB<bpc; iB++){
for(int i=0; i<b; i++){
for(int jB=C->Brow[iB]; jB<C->Brow[iB+1]; jB++){
int bcol = C->Bcol[jB];
int blk = iB*bpr + bcol;
for(int j=Crow_dist1[blk][i]; j<Crow_dist1[blk][i+1]; j++){
C->LLcol[nzb_cnt] = bcol*b + Ccol_dist1[blk][j];
nzb_cnt++;
}
if( i==(b-1) ){
free(Crow_dist1[blk]);
free(Ccol_dist1[blk]);
}    
}
C->LLrow[iB*b + i + 1] = nzb_cnt;
}
}
clock_gettime(CLOCK_MONOTONIC, &rbE);
printf("rebuild: %lf\n",timeConv(diff(rbS,rbE)));
printf("nnz=%d\n",C->LLrow[C->n]);
}
void BSpGEMM_dor(CSRbCSR *A,CSRbCSR *B, CSRbCSR *C, int b){
struct timespec mS,mE,rbS,rbE;
int bpc = A->n / b;
int bpr = B->m / b;
int NB  = bpc*bpr; 
int init_size = A->nnz;
int  *Ccol_size1 = malloc(NB*sizeof(int)); 
int  *Ccol_size2 = malloc(NB*sizeof(int));
int **Ccol_dist1 = malloc(NB*sizeof(int*));
int **Ccol_dist2 = malloc(NB*sizeof(int*));
int **Crow_dist1 = malloc(NB*sizeof(int*));
int **Crow_dist2 = malloc(NB*sizeof(int*));
bool is_nzb;
bool *first_time = malloc(NB*sizeof(bool));
bool *cnt        = calloc(NB,sizeof(bool));
for(int i=0; i<NB; i++) first_time[i]=1;
C->n = A->n;
C->bpc = bpc;
C->bpr = bpr;
int CBsize = A->nnz;
C->Bcol = malloc(CBsize*sizeof(int));
C->Brow = calloc((C->bpc+1),sizeof(int));
int Vsize = A->nnzb;
int *Vcol = malloc(Vsize*sizeof(int));
int *Vrow = calloc((bpc+1),sizeof(int));
int *temp;
clock_gettime(CLOCK_MONOTONIC, &mS);
for(int i=0; i<bpc; i++){
for(int jj=A->Brow[i]; jj<A->Brow[i+1]; jj++){
int j = A->Bcol[jj];
for(int k=B->Brow[j]; k<B->Brow[j+1]; k++){
int Aptr = b*jj;
int Bptr = b*k;
int blk = i*bpr + B->Bcol[k];
if(first_time[blk]){
first_time[blk]=0;
Ccol_size1[blk] = init_size;
Ccol_size2[blk] = init_size;
Ccol_dist1[blk] = malloc(init_size*sizeof(int));
Ccol_dist2[blk] = malloc(init_size*sizeof(int));
Crow_dist1[blk] = calloc((b+1),sizeof(int));
Crow_dist2[blk] = calloc((b+1),sizeof(int));
insertCSR_mv(&Vcol,Vrow,B->Bcol[k],&Vsize,i,bpc);
}
is_nzb = SpGEMM_dor(A->LLcol,&A->LLrow[Aptr],b,
B->LLcol,&B->LLrow[Bptr],b,
&Ccol_dist1[blk],Crow_dist1[blk],&Ccol_size1[blk],
Ccol_dist2[blk],Crow_dist2[blk]);
if(is_nzb) insertCSR_mv(&C->Bcol,C->Brow,B->Bcol[k],&CBsize,i,C->bpc);
temp = Ccol_dist2[blk]; 
Ccol_dist2[blk] = Ccol_dist1[blk];
Ccol_dist1[blk] = temp;
temp = Crow_dist2[blk];
Crow_dist2[blk] = Crow_dist1[blk];
Crow_dist1[blk] = temp;
swap(&Ccol_size1[blk],&Ccol_size2[blk]);
}
}
for(int jB=Vrow[i]; jB<Vrow[i+1]; jB++){ 
int blk= i*bpr + Vcol[jB];
if(!first_time[blk]){
int *temp_col_ptr;
int *temp_row_ptr;
free(Ccol_dist1[blk]);
free(Crow_dist1[blk]);
}
}
}
clock_gettime(CLOCK_MONOTONIC, &mE);
printf("multiply: %lf\n",timeConv(diff(mS,mE)));
clock_gettime(CLOCK_MONOTONIC, &rbS);
C->nnz=0;
for(int i=0; i<bpc; i++){
for(int j=C->Brow[i]; j<C->Brow[i+1]; j++){
int blk = i*bpr + C->Bcol[j];
C->nnz += Crow_dist2[blk][b];
}
}
C->LLcol = malloc(C->nnz*sizeof(int));
C->LLrow = malloc((C->n+1)*sizeof(int));
C->LLrow[0] = 0;
int nzb_cnt = 0;
for(int iB=0; iB<bpc; iB++){
for(int i=0; i<b; i++){
for(int jB=C->Brow[iB]; jB<C->Brow[iB+1]; jB++){
int bcol = C->Bcol[jB];
int blk = iB*bpr + bcol;
for(int j=Crow_dist2[blk][i]; j<Crow_dist2[blk][i+1]; j++){
C->LLcol[nzb_cnt] = bcol*b + Ccol_dist2[blk][j];
nzb_cnt++;
}
if( i==(b-1) ){
free(Crow_dist2[blk]);
free(Ccol_dist2[blk]);
}    
}
C->LLrow[iB*b + i + 1] = nzb_cnt;
}
}
clock_gettime(CLOCK_MONOTONIC, &rbE);
printf("rebuild: %lf\n",timeConv(diff(rbS,rbE)));
printf("nnz=%d\n",C->LLrow[C->n]);
}
void BSpGEMM_dor_n2(CSRbCSR *A,CSRbCSR *B, CSRbCSR *C, int b){
struct timespec mS,mE,rbS,rbE;
int bpc = A->n / b;
int bpr = B->m / b;
int NB  = bpc*bpr; 
int init_size = A->nnz;
int  *Ccol_size1 = malloc(NB*sizeof(int)); 
int  *Ccol_size2 = malloc(NB*sizeof(int));
int **Ccol_dist1 = malloc(NB*sizeof(int*));
int **Ccol_dist2 = malloc(NB*sizeof(int*));
int **Crow_dist1 = malloc(NB*sizeof(int*));
int **Crow_dist2 = malloc(NB*sizeof(int*));
bool is_nzb;
bool *first_time = malloc(NB*sizeof(bool));
bool *cnt        = calloc(NB,sizeof(bool));
for(int i=0; i<NB; i++) first_time[i]=1;
C->n = A->n;
C->bpc = bpc;
C->bpr = bpr;
int CBsize = A->nnz;
C->Bcol = malloc(CBsize*sizeof(int));
C->Brow = calloc((C->bpc+1),sizeof(int));
int Vsize = A->nnzb;
int *Vcol = malloc(Vsize*sizeof(int));
int *Vrow = calloc((bpc+1),sizeof(int));
int *temp;
clock_gettime(CLOCK_MONOTONIC, &mS);
for(int i=0; i<bpc; i++){
for(int jj=A->Brow[i]; jj<A->Brow[i+1]; jj++){
int j = A->Bcol[jj];
for(int k=B->Brow[j]; k<B->Brow[j+1]; k++){
int Aptr = b*jj;
int Bptr = b*k;
int blk = i*bpr + B->Bcol[k];
if(first_time[blk]){
first_time[blk]=0;
Ccol_size1[blk] = init_size;
Ccol_size2[blk] = init_size;
Ccol_dist1[blk] = malloc(init_size*sizeof(int));
Ccol_dist2[blk] = malloc(init_size*sizeof(int));
Crow_dist1[blk] = calloc((b+1),sizeof(int));
Crow_dist2[blk] = calloc((b+1),sizeof(int));
insertCSR_mv(&Vcol,Vrow,B->Bcol[k],&Vsize,i,bpc);
}
is_nzb = SpGEMM_dor(A->LLcol,&A->LLrow[Aptr],b,
B->LLcol,&B->LLrow[Bptr],b,
&Ccol_dist1[blk],Crow_dist1[blk],&Ccol_size1[blk],
Ccol_dist2[blk],Crow_dist2[blk]);
if(is_nzb) insertCSR_mv(&C->Bcol,C->Brow,B->Bcol[k],&CBsize,i,C->bpc);
temp = Ccol_dist2[blk]; 
Ccol_dist2[blk] = Ccol_dist1[blk];
Ccol_dist1[blk] = temp;
temp = Crow_dist2[blk];
Crow_dist2[blk] = Crow_dist1[blk];
Crow_dist1[blk] = temp;
swap(&Ccol_size1[blk],&Ccol_size2[blk]);
}
}
for(int jB=Vrow[i]; jB<Vrow[i+1]; jB++){ 
int blk= i*bpr + Vcol[jB];
if(!first_time[blk]){
int *temp_col_ptr;
int *temp_row_ptr;
free(Ccol_dist1[blk]);
free(Crow_dist1[blk]);
}
}
}
clock_gettime(CLOCK_MONOTONIC, &mE);
printf("multiply: %lf\n",timeConv(diff(mS,mE)));
clock_gettime(CLOCK_MONOTONIC, &rbS);
C->nnz=0;
for(int i=0; i<bpc; i++){
for(int j=C->Brow[i]; j<C->Brow[i+1]; j++){
int blk = i*bpr + C->Bcol[j];
C->nnz += Crow_dist2[blk][b];
}
}
C->LLcol = malloc(C->nnz*sizeof(int));
C->LLrow = malloc((C->n+1)*sizeof(int));
C->LLrow[0] = 0;
int nzb_cnt = 0;
for(int iB=0; iB<bpc; iB++){
for(int i=0; i<b; i++){
for(int jB=C->Brow[iB]; jB<C->Brow[iB+1]; jB++){
int bcol = C->Bcol[jB];
int blk = iB*bpr + bcol;
for(int j=Crow_dist2[blk][i]; j<Crow_dist2[blk][i+1]; j++){
C->LLcol[nzb_cnt] = bcol*b + Ccol_dist2[blk][j];
nzb_cnt++;
}
if( i==(b-1) ){
free(Crow_dist2[blk]);
free(Ccol_dist2[blk]);
}    
}
C->LLrow[iB*b + i + 1] = nzb_cnt;
}
}
clock_gettime(CLOCK_MONOTONIC, &rbE);
printf("rebuild: %lf\n",timeConv(diff(rbS,rbE)));
printf("nnz=%d\n",C->LLrow[C->n]);
}
void BSpGEMM_dor_nosort(CSRbCSR *A,CSRbCSR *B, CSRbCSR *C, int b){
struct timespec mS,mE,rbS,rbE;
int bpc = A->n / b;
int bpr = B->m / b;
int NB  = bpc*bpr; 
int init_size = A->nnz;
int  *Ccol_size1 = malloc(NB*sizeof(int)); 
int  *Ccol_size2 = malloc(NB*sizeof(int));
int **Ccol_dist1 = malloc(NB*sizeof(int*));
int **Ccol_dist2 = malloc(NB*sizeof(int*));
int **Crow_dist1 = malloc(NB*sizeof(int*));
int **Crow_dist2 = malloc(NB*sizeof(int*));
bool is_nzb;
bool *first_time = malloc(NB*sizeof(bool));
bool *cnt        = calloc(NB,sizeof(bool));
for(int i=0; i<NB; i++) first_time[i]=1;
C->n = A->n;
C->bpc = bpc;
C->bpr = bpr;
int CBsize = A->nnz;
C->Bcol = malloc(CBsize*sizeof(int));
C->Brow = calloc((C->bpc+1),sizeof(int));
int Vsize = A->nnzb;
int *Vcol = malloc(Vsize*sizeof(int));
int *Vrow = calloc((bpc+1),sizeof(int));
int *temp,temp_size;
clock_gettime(CLOCK_MONOTONIC, &mS);
for(int i=0; i<bpc; i++){
for(int jj=A->Brow[i]; jj<A->Brow[i+1]; jj++){
int j = A->Bcol[jj];
for(int k=B->Brow[j]; k<B->Brow[j+1]; k++){
int Aptr = b*jj;
int Bptr = b*k;
int blk = i*bpr + B->Bcol[k];
if(first_time[blk]){
first_time[blk]=0;
Ccol_size1[blk] = init_size;
Ccol_size2[blk] = init_size;
Ccol_dist1[blk] = malloc(init_size*sizeof(int));
Ccol_dist2[blk] = malloc(init_size*sizeof(int));
Crow_dist1[blk] = calloc((b+1),sizeof(int));
Crow_dist2[blk] = calloc((b+1),sizeof(int));
insertCSR_mv(&Vcol,Vrow,B->Bcol[k],&Vsize,i,bpc);
}
is_nzb = SpGEMM_dor_nosort(A->LLcol,&A->LLrow[Aptr],b,
B->LLcol,&B->LLrow[Bptr],b,
&Ccol_dist1[blk],Crow_dist1[blk],&Ccol_size1[blk],
Ccol_dist2[blk],Crow_dist2[blk]);
if(is_nzb) insertCSR_mv(&C->Bcol,C->Brow,B->Bcol[k],&CBsize,i,C->bpc);
temp = Ccol_dist2[blk]; 
Ccol_dist2[blk] = Ccol_dist1[blk];
Ccol_dist1[blk] = temp;
temp = Crow_dist2[blk];
Crow_dist2[blk] = Crow_dist1[blk];
Crow_dist1[blk] = temp;
swap(&Ccol_size1[blk],&Ccol_size2[blk]);
}
}
for(int jB=Vrow[i]; jB<Vrow[i+1]; jB++){ 
int blk= i*bpr + Vcol[jB];
free(Ccol_dist1[blk]);
free(Crow_dist1[blk]);
}
for(int jB=C->Brow[i]; jB<C->Brow[i+1]; jB++){ 
int blk= i*bpr + C->Bcol[jB];
for(int jj=0; jj<b; jj++){
quickSort(Ccol_dist2[blk],Crow_dist2[blk][jj],Crow_dist2[blk][jj+1]-1);
}
}
}
clock_gettime(CLOCK_MONOTONIC, &mE);
printf("multiply: %lf\n",timeConv(diff(mS,mE)));
clock_gettime(CLOCK_MONOTONIC, &rbS);
C->nnz=0;
for(int i=0; i<bpc; i++){
for(int j=C->Brow[i]; j<C->Brow[i+1]; j++){
int blk = i*bpr + C->Bcol[j];
C->nnz += Crow_dist2[blk][b];
}
}
C->LLcol = malloc(C->nnz*sizeof(int));
C->LLrow = malloc((C->n+1)*sizeof(int));
C->LLrow[0] = 0;
int nzb_cnt = 0;
for(int iB=0; iB<bpc; iB++){
for(int i=0; i<b; i++){
for(int jB=C->Brow[iB]; jB<C->Brow[iB+1]; jB++){
int bcol = C->Bcol[jB];
int blk = iB*bpr + bcol;
for(int j=Crow_dist2[blk][i]; j<Crow_dist2[blk][i+1]; j++){
C->LLcol[nzb_cnt] = bcol*b + Ccol_dist2[blk][j];
nzb_cnt++;
}
if( i==(b-1) ){
free(Crow_dist2[blk]);
free(Ccol_dist2[blk]);
}    
}
C->LLrow[iB*b + i + 1] = nzb_cnt;
}
}
clock_gettime(CLOCK_MONOTONIC, &rbE);
printf("rebuild: %lf\n",timeConv(diff(rbS,rbE)));
printf("nnz=%d\n",C->LLrow[C->n]);
}
int test_bsp(int argc, char const *argv[]){
struct timespec bS,bE,nbS,nbE;
int b = atoi(argv[2]);
uint32_t *Arow,*Acol;
int An,Am,Annz;
readCOO(argv[1],&Arow,&Acol,&An,&Am,&Annz);
printf("n=%d, nnz=%d\n",An,Annz);
CSRbCSR *A = malloc(sizeof(CSRbCSR));
CSRbCSR *C = malloc(sizeof(CSRbCSR));
clock_gettime(CLOCK_MONOTONIC, &bS);
csr2bcsr(Arow,Acol,An,Am,Annz,b,A,0);
clock_gettime(CLOCK_MONOTONIC, &bE);
printf("build: %lf\n",timeConv(diff(bS,bE)));
BSpGEMM_dor_nosort(A,A,C,b);
int Csize = 1000;
int *nCrow = calloc((An+1),sizeof(int));
int *nCcol = malloc(Csize*sizeof(int));
clock_gettime(CLOCK_MONOTONIC, &nbS);
SpGEMM_d(Acol,Arow,An,Acol,Arow,An,&nCcol,nCrow,&Csize);
clock_gettime(CLOCK_MONOTONIC, &nbE);
printf("non-blocked: %lf\n",timeConv(diff(nbS,nbE)));
printf("nnz=%d\n",nCrow[An]);
}
void test_d(int argc, char const *argv[]){
struct timespec bS,bE,nbS,nbE;
int b = atoi(argv[2]);
uint32_t *Arow,*Acol;
int An,Am,Annz;
readCOO(argv[1],&Arow,&Acol,&An,&Am,&Annz);
int Csize = 1000;
int *nCrow = calloc((An+1),sizeof(int));
int *nCcol = malloc(Csize*sizeof(int));
clock_gettime(CLOCK_MONOTONIC, &nbS);
SpGEMM_d(Acol,Arow,An,Acol,Arow,An,&nCcol,nCrow,&Csize);
clock_gettime(CLOCK_MONOTONIC, &nbE);
printf("non-blocked: %lf\n",timeConv(diff(nbS,nbE)));
printf("nnz=%d\n",nCrow[An]);
printCSR(nCrow,nCcol,An,An,b);
}
void test_dor(int argc, char const *argv[]){
struct timespec bS,bE,nbS,nbE;
int b = atoi(argv[2]);
uint32_t *Arow,*Acol;
int An,Am,Annz;
readCOO(argv[1],&Arow,&Acol,&An,&Am,&Annz);
int Csize = 1000;
int *nCrow = calloc((An+1),sizeof(int));
int *nCcol = malloc(Csize*sizeof(int));
int Dcol[] = {3,8,3,4,7,5,6,7,0};
int Drow[] = {0,2,5,8,9,9,9,9,9,9};
clock_gettime(CLOCK_MONOTONIC, &nbS);
SpGEMM_dor(Acol,Arow,An,Acol,Arow,An,&nCcol,nCrow,&Csize,Acol,Arow);
clock_gettime(CLOCK_MONOTONIC, &nbE);
printf("non-blocked: %lf\n",timeConv(diff(nbS,nbE)));
printf("nnz=%d\n",nCrow[An]);
}
void test_mask(int argc, char const *argv[]){
struct timespec bS,bE,nbS,nbE;
int b = atoi(argv[2]);
uint32_t *Arow,*Acol;
int An,Am,Annz;
readCOO(argv[1],&Arow,&Acol,&An,&Am,&Annz);
int Csize = 1000;
int *nCrow = calloc((An+1),sizeof(int));
int *nCcol = malloc(Csize*sizeof(int));
int Dcol[] = {0};
int *Drow = calloc(An+1,sizeof(int));
clock_gettime(CLOCK_MONOTONIC, &nbS);
SpGEMM_dor_masked(Acol,Arow,An,Acol,Arow,An,Acol,Arow,&nCcol,nCrow,&Csize,Dcol,Drow);
clock_gettime(CLOCK_MONOTONIC, &nbE);
printf("non-blocked: %lf\n",timeConv(diff(nbS,nbE)));
printf("nnz=%d\n",nCrow[An]);
}
int main(int argc, char const *argv[])
{
if(argc!=3){
printf("usage: ./BSpGEMM matrix block-size\n");
exit(1);
}
test_bsp(argc,argv);
return 0;
}
