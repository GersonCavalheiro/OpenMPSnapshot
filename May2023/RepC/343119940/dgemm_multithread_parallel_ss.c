#include <stdio.h>
#include <time.h>
#include <xmmintrin.h>
#include <omp.h>
#define UNROLL 4
#define BLOCKSIZE 32
#define Nmatriz 4096
#define Nmax (Nmatriz * Nmatriz)
float a[(int)Nmax], b[(int)Nmax], c[(int)Nmax];
void inicializa(){
int i;
time_t semilla = (long int) 223456;  
srand48((long int)semilla);
for(i = 0; i < Nmax; i++){ 
a[i] = (float) i; 
b[i] = a[i];
c[i] = 0;
}
}
void dgemm_blocking (int n, int si, int sj, int sk, float* A, float* B, float* C){
int i,j,k,x;
for ( i = si; i < si+BLOCKSIZE; i+=UNROLL*4 )
for ( j = sj; j < sj+BLOCKSIZE; j++ ) {
__m128 c[4];
for ( x = 0; x < UNROLL; x++ )
c[x] = _mm_load_ps(C+i+x*4+j*n);
for( k = sk; k < sk+BLOCKSIZE; k++ ) {
__m128 b = _mm_load_ps1(B+k+j*n); 
for ( x = 0; x < UNROLL; x++ )
c[x] = _mm_add_ps(c[x], 
_mm_mul_ps(_mm_load_ps(A+i+x*4+k*n), b ));
}
for ( x = 0; x < UNROLL; x++ )
_mm_store_ps(C+i+x*4+j*n, c[x]); 
}
}
void dgemm (int n, float* A, float* B, float* C){
int i,j,k;
#pragma omp parallel for private (i,k)
for ( j = 0; j < n; j += BLOCKSIZE )
for ( i = 0; i < n; i += BLOCKSIZE )
for ( k = 0; k < n; k += BLOCKSIZE )
dgemm_blocking (n, i, j, k, A, B, C);
}
main(){
double tiempoInicio, tiempoFinal;
double tiempo, MFlops;
int i,j;
double dNmatriz = (double) Nmatriz;
double Nops = 2 * dNmatriz * dNmatriz * dNmatriz;
tiempoInicio = omp_get_wtime();
inicializa();
tiempoFinal = omp_get_wtime();
tiempo = ((float)tiempoFinal - (float)tiempoInicio) / (float)CLOCKS_PER_SEC;
printf("inicializa una matriz de %i x %i floats: %7.4f seg\n", Nmatriz, Nmatriz, tiempo);
tiempoInicio = omp_get_wtime();
dgemm (Nmatriz, a, b, c);
tiempoFinal = omp_get_wtime();
tiempo = tiempoFinal - tiempoInicio;
printf("dgemm_secuencial: %7.4f seg,", tiempo);
printf(" %7.4f MFlops\n", (tiempo>0?(double)(Nops/(tiempo*1e6)):0));
}
