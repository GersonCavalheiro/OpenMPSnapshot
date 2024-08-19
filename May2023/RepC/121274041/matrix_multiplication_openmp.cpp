#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
using namespace std;
int thread_count = 1;        
void multiply(double ** &mat1,double ** &mat2,double ** &res,int m,int n,int o);
void transpose(double ** &res, int m, int o);
void matAlloc(double ** &mat, int row, int col);
void matClean(double ** &mat, int row);
void matFill(double ** &mat, int row, int col);
int main(int argc, char *argv[]) {
if(argc != 2) {
cout << "Usage: " << argv[0] << " (number of threads)\n";
exit(-1);
}
thread_count = atoi(argv[1]);
double **matrix1, **matrix2, **result;       
int m = 4000, n = 3000, o = 3200;            
timeval start, end;                          
matAlloc(matrix1, m, n);
matAlloc(matrix2, n, o);
matAlloc(result , m, o);
matFill(matrix1, m, n);
matFill(matrix2, n, o);
gettimeofday(&start, NULL);                  
#pragma omp parallel num_threads(thread_count)
multiply(matrix1, matrix2, result, m, n, o);
gettimeofday(&end, NULL);                   
cout << "Elapsed Multiply: " << ((end.tv_sec-start.tv_sec)*1000000+end.tv_usec-start.tv_usec)/1000000.0 << endl;
gettimeofday(&start, NULL);                  
transpose(result, m, o);                     
gettimeofday(&end, NULL);                    
cout << "Elapsed Transpose: " << ((end.tv_sec-start.tv_sec)*1000000+end.tv_usec-start.tv_usec)/1000000.0 << endl;
matClean(matrix1, m);
matClean(matrix2, n);
matClean(result , o);
return 0;
}
void multiply(double ** &mat1,double ** &mat2,double ** &res,int m,int n,int o) {
#pragma omp for
for(int i=0; i<m; i++) {
for(int j=0; j<o; j++) {
double sum = 0.0;
for(int k=0; k<n; k++) {
sum += mat1[i][k]*mat2[k][j];
}
res[i][j] = sum;
}
}
}
void transpose(double ** &res, int row, int col) {
double ** temp; matAlloc(temp, col, row);
#pragma omp parallel for num_threads(thread_count)
for(int i=0; i<row; i++) {
for(int j=0; j<col; j++) {
temp[j][i] = res[i][j];
}
}
matClean(res, row);
res = temp;
}
void matAlloc(double ** &mat, int row, int col) {
mat = new double*[row];
for(int i=0; i<row; i++) {
mat[i] = new double[col];
}
}
void matClean(double ** &mat, int row) {
for(int i=0; i<row; i++) {
delete [] mat[i];
}
delete [] mat;
}
void matFill(double ** &mat, int row, int col) {
srand(time(NULL));
for(int i=0; i<row; i++) {
for(int j=0; j<col; j++){
mat[i][j] = (rand() % 10000)/10000.0;
}
}
}
