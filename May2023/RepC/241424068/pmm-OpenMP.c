#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define VECTOR_DYNAMIC  
int main(int argc, char **argv)  {
int i, j, h;
if(argc < 2){
printf("Faltan nº componentes del vector\n");
exit(-1);
}
unsigned int n = atoi(argv[1]);
if(n < 2){
fprintf(stderr, "El tamaño no puede ser menor a 2");
exit(-1);
}
#ifdef VECTOR_DYNAMIC
double **m1, **m2, **resultado;
m1 = (double**) malloc(n*sizeof(double*));
m2 = (double**) malloc(n*sizeof(double*));
resultado = (double**) malloc(n*sizeof(double*));
for(i=0; i<n; i++){
m1[i] = (double*) malloc(n*sizeof(double));
m2[i] = (double*) malloc(n*sizeof(double));
resultado[i] = (double*) malloc(n*sizeof(double));
}
#endif
double t1, t2;
#pragma omp parallel shared(m1,m2,resultado) private(i,j,h)
{
#pragma omp for schedule(runtime)
for(i=0; i<n; i++)
for(j=0; j<n; j++){
m1[i][j] = 5;
m2[i][j] = 5;
resultado[i][j] = 0;
}
#pragma omp single
{
t1 = omp_get_wtime();
}
#pragma omp for schedule(runtime)
for(i=0; i<n; i++)
for(j=0; j<n; j++)
for(h=0; h<n; h++){
resultado[i][j] += m1[i][h]*m1[h][j];
}
#pragma omp single
{
t2 = omp_get_wtime() - t1;
}
}
printf("Tiempo: %f\n Tamaño: %i\n\n", t2, n);
if(n <= 10){
for(i=0; i<n; i++){
for(j=0; j<n; j++)
printf("%0.2f  ", resultado[i][j]);
printf("\n");
}
}
else{
printf("Componente [0,0]: %0.2f\n Componente [N-1,N-1]: %0.2f\n", resultado[0][0], resultado[n-1][n-1]);
}
#ifdef VECTOR_DYNAMIC
for(i=0; i<n; i++){
free(m1[i]);
free(m2[i]);
free(resultado[i]);
}
free(m1);
free(m2);
free(resultado);
#endif
return 0;
}
