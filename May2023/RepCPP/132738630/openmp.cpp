#include <iostream>
#include <chrono>
#include <cstring>
#include <omp.h>
#define DEBUG 0

using namespace std;


extern float** createRandomMatrix(long, long, bool);
extern float** createIdentityMatrix(long);
extern float** createEmptyMatrix(long);
extern void print_matrix(float**, long, char*);
extern void saveTimeToFile(long, float, char*);
extern bool multipliedMatrixIsCorrect(float**, float**, float**, long);


void mat_inv(float**, float**, long);
void mat_mul(float**, float**, float**, long);

int main(int argc, char **argv){
long min_dim, max_dim, step, dim; 
chrono::high_resolution_clock::time_point start, finish; 
chrono::duration<double> elapsed; 
float **A, **B, **C; 
float **D, **M; 

omp_set_num_threads(4); 

if(argc != 4){
cout << "Usage: " << argv[0] << " [min_dim] [max_dim] [step]" << endl;
return -1;
}

min_dim = strtol(argv[1], NULL, 10);
max_dim = strtol(argv[2], NULL, 10)+1; 
step = strtol(argv[3], NULL, 10);


for(dim=min_dim;dim<max_dim;dim+=step){

A = createRandomMatrix(dim, dim, true); 
B = createRandomMatrix(dim, dim, false); 
C = createEmptyMatrix(dim);
D = createIdentityMatrix(dim);

M = new float*[dim];
for (int h = 0; h < dim; h++){
M[h] = new float[dim];
for (int w = 0; w < dim; w++)
M[h][w] = A[h][w];
}

if(DEBUG){
print_matrix(A,dim,"A ");
print_matrix(B,dim,"B ");
}


start = chrono::high_resolution_clock::now(); 


mat_mul(A,B,C,dim); 


finish = chrono::high_resolution_clock::now(); 

if(DEBUG){
print_matrix(C,dim,"C ");
bool correct = multipliedMatrixIsCorrect(A,B,C,dim);
if(!correct){
cout << "Multiplied matrix is not correct, aborting..." << endl;
return -1;
}
}

elapsed = finish - start; 

if(DEBUG) cout << "MUL: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;

saveTimeToFile(dim, elapsed.count(), "csv/multiplication_OpenMP.csv");


start = chrono::high_resolution_clock::now(); 


mat_inv(M,D,dim); 


finish = chrono::high_resolution_clock::now(); 


if(DEBUG){
print_matrix(M,dim,"M ");
print_matrix(D,dim,"D ");
bool correct = multipliedMatrixIsCorrect(A,D,M,dim);
if(!correct){
cout << "Multiplied matrix is not correct, aborting..." << endl;
return -1;
}
}

elapsed = finish - start; 

if(DEBUG) cout << "INV: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;

saveTimeToFile(dim, elapsed.count(), "csv/inversion_OpenMP.csv");

free(A);
free(B);
free(C);
free(D);
free(M);
}

return 0;
}

void mat_inv(float **M, float **D, long dim){
double s;
int i,j;

for(int piv=0; piv<dim; piv++){ 
#pragma omp parallel for private(i,j,s)
for(int i=piv+1; i<dim; i++){ 
s = (double)M[i][piv]/M[piv][piv];
for(int j=0;j<dim;j++){ 
D[i][j] -= (float)s*D[piv][j];
if(j>=piv) M[i][j] -= s*M[piv][j];
}
}
}

for(int piv=dim-1; piv>=0; piv--){ 
#pragma omp parallel for private(i,j,s)
for(int i=0; i<piv; i++){ 
s = (double)M[i][piv]/M[piv][piv];
for(int j=0;j<dim;j++){ 
D[i][j] -= (float)s*D[piv][j];
if(j<=piv) M[i][j] -= s*M[piv][j];
}
}
}

for(int i=0;i<dim;i++){
s = M[i][i];
#pragma omp parallel for private(j)
for(int j=0;j<dim;j++){
D[i][j]/=s;
M[i][j]/=s;
}
}

return;
}

void mat_mul(float **A,float **B, float** prodotto, long n){

int i,j,k;
#pragma omp parallel for private(i,j,k)
for (i = 0; i < n; i++) {
for (j = 0; j < n; j++) {
prodotto[i][j] = 0;
for (k = 0; k < n; k++) {
prodotto[i][j] += A[i][k] * B[k][j];
}
}
}

return;
}
