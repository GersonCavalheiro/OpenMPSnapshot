#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../timer.h"
struct LinearSystem
{
int n, m;                           
double **A;                         
double **U;                         
double *b;                          
double *b_ori;                      
double *x;                          
}typedef LinearSystem;
void error (const char *msg)
{
printf("%s\n",msg);
exit(EXIT_FAILURE);
}
void printMatrix (const char name[], double **A, int n, int m)
{   
printf("===== %s =====\n",name);
for (int i = 0; i < n; i++)
{
for (int j = 0; j < m; j++)
printf("%lf ",A[i][j]);
printf("\n");
}
printf("\n");
}
void printVector (const char name[], double *b, int n)
{   
printf("===== %s =====\n",name);
for (int i = 0; i < n; i++)
printf("%lf\n",b[i]);
printf("\n");
}
LinearSystem* newLinearSystem (int argc, char *argv[])
{
int i, j, k, p;
double v;
char *buffer = (char*)malloc(sizeof(char)*50);
size_t bufsize = 50;
size_t chars;
FILE *in;
LinearSystem *ls = (LinearSystem*)malloc(sizeof(LinearSystem));
in = fopen(argv[2],"r");
chars = getline(&buffer,&bufsize,in);
if (chars <= 0) error("Reading file");
if (!fscanf(in,"%d %d %d",&ls->n,&ls->m,&k)) error("Reading file");
ls->A = (double**)malloc(sizeof(double*)*ls->n);
ls->A[0] = (double*)calloc(ls->n*ls->m,sizeof(double));
ls->U = (double**)malloc(sizeof(double*)*ls->n);
ls->U[0] = (double*)calloc(ls->n*ls->m,sizeof(double));
for (i = 1; i < ls->n; i++)
{   
ls->A[i] = ls->A[0] + i*ls->m;
ls->U[i] = ls->U[0] + i*ls->m;
} 
for (p = 0; p < k; p++)
{
if (!fscanf(in,"%d %d %lf",&i,&j,&v)) error("Reading file");
i--; j--;
ls->A[i][j] = v;
ls->U[i][j] = v;
}
fclose(in);
in = fopen(argv[3],"r");
chars = getline(&buffer,&bufsize,in);
if (chars <= 0) error("Reading file");
if (!fscanf(in,"%d %d",&ls->n,&k)) error("Reading file");
ls->b = (double*)calloc(ls->n,sizeof(double));
ls->b_ori = (double*)calloc(ls->n,sizeof(double));
for (i = 0; i < ls->n; i++)
{
if (!fscanf(in,"%lf",&v)) error("Reading file");
ls->b[i] = v; ls->b_ori[i] = v;
}
ls->x = (double*)calloc(ls->m,sizeof(double));
fclose(in);
free(buffer);
return ls;
}
void BackSubstitution_Column (double **A, double *b, double *x, int n)
{
int row, col;
#pragma omp parallel for
for (row = 0; row < n; row++)
x[row] = b[row];
for (col = n-1; col >= 0; col--)
{
x[col] /= A[col][col];
#pragma omp parallel for
for (row = 0; row < col; row++)
x[row] -= A[row][col]*x[col];
}
}
void swap_rows (double **U, double *b, int i, int j)
{
double *aux = U[i];
U[i] = U[j];
U[j] = aux;
double aux2;
aux2 = b[i];
b[i] = b[j];
b[j] = aux2;
}
int max_col (double **U, int k, int n)
{
int maxIndex = k;
double maxValue = U[k][k];
for (int i = k; i < n; i++)
{
if (U[i][k] > maxValue)
{
maxValue = U[i][k];
maxIndex = i;
}
}
return maxIndex;
}
void forwardElimination (double **U, double *b, int n, int m)
{
int r;
double *l = (double*)calloc(n,sizeof(double));
for (int k = 0; k < n-1; k++)
{
r = max_col(U,k,n);
if (r != k) swap_rows(U,b,k,r);
#pragma omp parallel for
for (int i = k+1; i < n; i++)
{
l[i] = U[i][k] / U[k][k];
for (int j = k+1; j < m; j++)
U[i][j] = U[i][j] - l[i]*U[k][j];
b[i] = b[i] - l[i]*b[k];
}
}
free(l);
}
int checkSystem (double **A, double *x, double *b, int n, int m)
{
double error = 0.0;
for (int i = 0; i < n; i++)
{
double sum = 0.0;
for (int j = 0; j < m; j++)
sum += A[i][j]*x[j];
error += pow(b[i]-sum,2);
}
printf("Error = %.10lf\n",sqrt(error));
if (error > 1.0e-03) return 0;
else                 return 1;
}
void freeLinearSystem (LinearSystem *ls)
{
free(ls->A[0]);
free(ls->U[0]);
free(ls->b);
free(ls->b_ori);
free(ls->x);
free(ls);
}
void Usage (const char program_name[])
{
printf("========================== PARALLEL GAUSS ELIMINATION ==========================\n");
printf("Usage:> %s <num_threads> <coef_matrix> <rhs_vector>\n",program_name);
printf("<coef_matrix> = Coeffient matrix\n");
printf("<rhs_vector> = Right-hand side vector\n");
printf("--------------------------------------------------------------------------------\n");
printf("** The input files must be in the MatrixMarket format, '.mtx'. **\n");
printf("** More info: http:
printf("================================================================================\n");
}
int main (int argc, char *argv[])
{
int num_threads;
double start, finish, elapsed;
if (argc-1 < 3)
{
Usage(argv[0]);
exit(EXIT_FAILURE);
}
num_threads = atoi(argv[1]);
LinearSystem *ls = newLinearSystem(argc,argv);
omp_set_num_threads(num_threads);
GET_TIME(start);
forwardElimination(ls->U,ls->b,ls->n,ls->m);
BackSubstitution_Column(ls->U,ls->b,ls->x,ls->n);
GET_TIME(finish);
if (checkSystem(ls->A,ls->x,ls->b_ori,ls->n,ls->m)) printf("[+] The solution is correct !\n");
else                                                printf("[-] The solution is NOT correct !\n");
elapsed = finish - start;
printf("Time elapsed = %.10lf s\n",elapsed);
freeLinearSystem(ls);
return 0;
}
