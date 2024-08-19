#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#define  MAX_COL 2048
#define  MAX_ROW 2048
#define TILE_SIZE 16
#define TRUE 1
#define FALSE 0

struct timezone {
int tz_minuteswest;     
int tz_dsttime;         
};

int ** create_matrix(int arraySizeX, int arraySizeY);
int ** initialize(int ** M, int value);
int ** transpose(int ** M);

int ** multiply(int ** A, int ** B);
int ** t_transp_multiply(int ** A, int ** B);
int ** t_openmp_seq_mutlitply(int ** A, int ** B);
int ** t_openmp_transp_mutlitply(int ** A, int ** B);
int ** t_tiling_multiply(int **A, int **B,int width, int height, int tile_size);
int ** t_openmp_tiling_multiply(int **A, int **B,int width, int height, int tile_size);
int ** t_tiling_transpose_multiply(int **A, int **B,int width, int height, int tile_size);
int ** tiling_3D(int ** A, int ** B,int width, int height,int tile_size);
int ** tiling_transpose_3D(int ** A, int ** B,int width, int height,int tile_size);


int gettimeofday(struct timeval *tv, struct timezone *tz);
static double rtclock();
int are_equal(int ** A, int ** B);

int main(int arg, char ** args){
double begin, end;

int  ** A = NULL;
int  ** B = NULL;
int  ** product = NULL;
int  ** product_init = NULL;

A = create_matrix(MAX_ROW, MAX_COL);
B = create_matrix(MAX_ROW, MAX_COL);

A = initialize(A, 1);
B = initialize(B, 1);

printf("Sequential multiplication :\n");
begin = rtclock();
product = multiply(A, B);
product_init=product;
end = rtclock();
printf("Multiplication ended, time elapsed : %.2f sec\n", (float)(end - begin)  / CLOCKS_PER_SEC);

printf("\nTransposition multiplication :\n");
begin = rtclock();
product = t_transp_multiply(A, B);
end = rtclock();
printf("Multiplication ended, time elapsed : %.2f sec\n", (float)(end - begin)  / CLOCKS_PER_SEC);
printf("Got the same result ? ");
if (are_equal(product_init, product))
printf(" YES.\n");
else printf(" NO.\n");


printf("\nOpenMP sequential optimization multiplication :\n");
begin = rtclock();
product = t_openmp_seq_mutlitply(A, B);
end = rtclock();
printf("Multiplication ended, time elapsed : %.2f sec\n", (float)(end - begin)  / CLOCKS_PER_SEC);
printf("Got the same result ? ");
if (are_equal(product_init, product))
printf(" YES.\n");
else printf(" NO.\n");


printf("\nOpenMP transposition optimization multiplication :\n");
begin = rtclock();
product = t_openmp_transp_mutlitply(A,B);
end = rtclock();
printf("Multiplication ended, time elapsed : %.2f sec\n", (float)(end - begin)  / CLOCKS_PER_SEC);
printf("Got the same result ? ");
if (are_equal(product_init, product))
printf(" YES.\n");
else printf(" NO.\n");


printf("\nTiling multiplication :\n");
begin = rtclock();
product = t_tiling_multiply(A,B,MAX_COL, MAX_ROW,TILE_SIZE);
end = rtclock();
printf("Multiplication ended, time elapsed : %.2f sec\n", (float)(end - begin)  / CLOCKS_PER_SEC);
printf("Got the same result ? ");
if (are_equal(product_init, product))
printf(" YES.\n");
else printf(" NO.\n");


printf("\nOpenMP tiling multiplication :\n");
begin = rtclock();
product = t_openmp_tiling_multiply(A, B,MAX_COL, MAX_ROW,TILE_SIZE);
end = rtclock();
printf("Multiplication ended, time elapsed : %.2f sec\n", (float)(end - begin)  / CLOCKS_PER_SEC);
printf("Got the same result ? ");
if (are_equal(product_init, product))
printf(" YES.\n");
else printf(" NO.\n");


printf("\nTiling transpose multiplication :\n");
begin = rtclock();
product = t_tiling_transpose_multiply(A, B,MAX_COL, MAX_ROW,TILE_SIZE);
end = rtclock();
printf("Multiplication ended, time elapsed : %.2f sec\n", (float)(end - begin)  / CLOCKS_PER_SEC);
printf("Got the same result ? ");
if (are_equal(product_init, product))
printf(" YES.\n");
else printf(" NO.\n");


printf("\nTiling 3D multiplication :\n");
begin = rtclock();
product = tiling_3D(A, B,MAX_COL, MAX_ROW,TILE_SIZE);
end = rtclock();
printf("Multiplication ended, time elapsed : %.2f sec\n", (float)(end - begin)  / CLOCKS_PER_SEC);
printf("Got the same result ? ");
if (are_equal(product_init, product))
printf(" YES.\n");
else printf(" NO.\n");


printf("\nTiling transpose 3D multiplication :\n");
begin = rtclock();
product = tiling_transpose_3D(A, B,MAX_COL, MAX_ROW,TILE_SIZE);
end = rtclock();
printf("Multiplication ended, time elapsed : %.2f sec\n", (float)(end - begin)  / CLOCKS_PER_SEC);
printf("Got the same result ? ");
if (are_equal(product_init, product))
printf(" YES.\n");
else printf(" NO.\n");

return 0;
}

int** create_matrix(int arraySizeX, int arraySizeY) {
int** array;
array = (int**) malloc(arraySizeX*sizeof(int*));
int i;
for (i = 0; i < arraySizeX; i++) {
array[i] = (int*) malloc(arraySizeY*sizeof(int));
}
return array;
}

int ** initialize(int ** M, int value){
int i,j;
for (i = 0;  i < MAX_ROW; i++) 
for ( j = 0;  j < MAX_ROW; j++) {
M[i][j] = value;
}
return M;
}


int ** multiply(int ** A, int ** B)
{
int ** C = create_matrix(MAX_ROW, MAX_COL);
C = initialize(C, 0);
int i,j,k;
for (i = 0; i < MAX_COL; i++)
for (j = 0;  j < MAX_ROW; j++)
for (k = 0;  k < MAX_ROW; k++)
C[i][j] += A[i][k] * B[k][j];
return C;
}

int ** transpose(int ** M){
int ** T = create_matrix(MAX_ROW, MAX_COL);
int i,j;
for (i = 0; i < MAX_COL; i++)
for (j = 0;  j < MAX_ROW; j++)
T[i][j] = M[j][i];
return T;
}


int ** t_transp_multiply(int ** A, int ** B){
int ** C = create_matrix(MAX_ROW, MAX_COL);
C = initialize(C, 0);
B = transpose(B);
int i,j,k;
for ( i = 0; i < MAX_COL; i++)
for (j = 0;  j < MAX_ROW; j++)
for (k = 0;  k < MAX_ROW; k++)
C[i][j] += A[i][k] * B[j][k];
return C;
}

int ** t_openmp_seq_mutlitply(int ** A, int ** B) {
int ** C = create_matrix(MAX_ROW, MAX_COL);
C = initialize(C, 0);
int i,j,k;
#pragma omp parallel for private(i,j,k)
for ( i = 0; i < MAX_ROW; i++) {
for (j = 0; j < MAX_COL; j++) {
for (k = 0; k < MAX_COL; k++) {
C[i][j] += A[i][k] * B[k][j];
}
}
}
return C;
}

int ** t_openmp_transp_mutlitply(int ** A, int ** B) {
int ** C = create_matrix(MAX_ROW, MAX_COL);
C = initialize(C, 0);
B = transpose(B);
int i,j,k;
#pragma omp parallel for private(i,j,k)
for ( i = 0; i < MAX_COL; i++)
for (j = 0;  j < MAX_ROW; j++)
for (k = 0;  k < MAX_ROW; k++)
C[i][j] += A[i][k] * B[j][k];
return C;
}



int ** t_tiling_multiply(int **A, int **B,int width, int height, int tile_size) {
int ** C = create_matrix(height, width);
int i,j,k,x,y;
for (i=0; i < height/tile_size; i++){
for (j=0; j < height/tile_size; j++){
for (k=tile_size*i; k<tile_size*i+tile_size; k++){
for (x=tile_size*j; x<tile_size*j+tile_size; x++){
C[k][x] = 0;
for (y = 0; y < height; y++){ 
C[k][x] += A[k][y] * B[y][x];
}
}
}
}
}   

return C;
}


int ** t_openmp_tiling_multiply(int **A, int **B,int width, int height, int tile_size) {
int ** C = create_matrix(height, width);
int i,j,k,x,y;
#pragma omp parallel for private(i,j,k,x,y)
for (i=0; i < height/tile_size; i++){
for (j=0; j < height/tile_size; j++){
for (k=tile_size*i; k<tile_size*i+tile_size; k++){
for (x=tile_size*j; x<tile_size*j+tile_size; x++){
C[k][x] = 0;
for (y = 0; y < height; y++){ 
C[k][x] += A[k][y] * B[y][x];
}
}
}
}
} 

return C;
}


int ** t_tiling_transpose_multiply(int **A, int **B,int width, int height, int tile_size) {
int ** C = create_matrix(height, width);
B = transpose(B);
int i,j,k,x,y;
for (i=0; i < height/tile_size; i++){
for (j=0; j < height/tile_size; j++){
for (k=tile_size*i; k<tile_size*i+tile_size; k++){
for (x=tile_size*j; x<tile_size*j+tile_size; x++){
C[k][x] = 0;
for (y = 0; y < height; y++ ){ 
C[k][x] += A[k][y] * B[x][y];
}
}
}
}
}   

return C;
}


int ** tiling_3D(int ** A, int ** B,int width, int height,int tile_size) {   
int ** C = create_matrix(height, width);
int i,j,i0,j0,k0,i1,j1,k1, num = height/tile_size;
C = initialize(C,0);

for(i0=0; i0<num; i0++)
{
for(j0=0; j0<num; j0++)
{
for(k0=0; k0<num; k0++)
{
for(i1=tile_size*i0; i1<tile_size*i0 + tile_size; i1++)
{
for(j1=tile_size*j0; j1<tile_size*j0 + tile_size; j1++)
{
for(k1=tile_size*k0; k1<tile_size*k0 +tile_size; k1++)
{
C[i1][j1] += A[i1][k1] * B[k1][j1];
}

}
}
}
}
}

return C;
}


int ** tiling_transpose_3D(int ** A, int ** B,int width, int height,int tile_size) {   
int ** C = create_matrix(height, width);
int i,j,i0,j0,k0,i1,j1,k1, num = height/tile_size;
B = transpose(B);
C = initialize(C,0);

for(i0=0; i0<num; i0++)
{
for(j0=0; j0<num; j0++)
{
for(k0=0; k0<num; k0++)
{
for(i1=tile_size*i0; i1<tile_size*i0 + tile_size; i1++)
{
for(j1=tile_size*j0; j1<tile_size*j0 + tile_size; j1++)
{
for(k1=tile_size*k0; k1<tile_size*k0 +tile_size; k1++)
{
C[i1][j1] += A[i1][k1] * B[j1][k1];
}

}
}
}
}
}

return C;
}



static double rtclock()
{
struct timeval Tp;
gettimeofday (&Tp, NULL);
return (Tp.tv_sec * 1.e6 + Tp.tv_usec);
}

int are_equal(int ** A, int ** B){

for (int i = 0; i < MAX_COL; i++)
for (int j = 0;  j < MAX_ROW; j++)
if (A[i][j] != B[i][j])
return FALSE;
return TRUE;
}
