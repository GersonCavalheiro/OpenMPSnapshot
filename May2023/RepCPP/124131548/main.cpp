
# include <stdlib.h>
# include <stdio.h>
#include <time.h>
# include <omp.h>

int main ( int argc, char *argv[] );
void mostrar_resultados ( int n_lo, int n_hi, int n_factor );
int numero_primo ( int n );
int numero_primo_paralelo ( int n );

int main ( int argc, char *argv[] )
{
int n_factor;
int n_max;
int n_min;

printf ( "\n" );
printf ( "Numeros primos con OPENMP\n" );

printf ( "\n" );
printf ( "  Numero de procesadores disponibles = %d\n", omp_get_num_procs ());
printf ( "  Numero de threads	= %d\n", omp_get_max_threads ());

n_min = 1;
n_max = 131072;
n_factor = 2;

mostrar_resultados ( n_min, n_max, n_factor );




printf ( "\n" );
printf ( " Fin del programa.\n" );
getchar();
return 0;
}

void mostrar_resultados ( int n_min, int n_max, int n_factor )
{
int n;
int primos;
double tiempo;
double tiempo_total;
clock_t t;
clock_t tt;

printf ( "\n" );
printf ( "  Se llama a la funcion numero_primo para contar los primos del 1 a N.\n" );
printf ( "\n" );
printf ( "         N       Pri          Tiempo\n" );
printf ( "\n" );

n = n_min;
tiempo_total = 0;
tt = 0;

while ( n <= n_max )
{
t = clock();

primos = numero_primo ( n );

t = clock() - t;

printf ( "  %8d  %8d  %14f\n", n, primos, ((double)t)/CLOCKS_PER_SEC );

n = n * n_factor;
tt = tt+t;
}
printf ( "Tiempo total = %14f\n", ((double)tt)/CLOCKS_PER_SEC);

printf ( "\n" );
printf ( "  Se llama a la funcion numero_primo_paralelo para contar los primos del 1 a N.\n" );
printf ( "\n" );
printf ( "         N       Pri          Tiempo\n" );
printf ( "\n" );

n = n_min;
tiempo_total = 0;

while ( n <= n_max )
{
tiempo = omp_get_wtime ( );

primos = numero_primo_paralelo ( n );

tiempo = omp_get_wtime ( ) - tiempo;

printf ( "  %8d  %8d  %14f\n", n, primos, tiempo );

n = n * n_factor;
tiempo_total = tiempo_total+tiempo;
}
printf ( "Tiempo total = %14f\n", tiempo_total);

return;
}

int numero_primo ( int n )
{
int i;
int j;
int primo;
int total = 0; 

for ( i = 2; i <= n; i++ )
{
primo = 1;

for ( j = 2; j < i; j++ )
{
if ( i % j == 0 )
{
primo = 0;
break;
}
}
total = total + primo;
}

return total;
}

int numero_primo_paralelo ( int n )
{
int i;
int j;
int primo;
int total = 0;

# pragma omp parallel \
shared ( n ) \
private ( i, j, primo )

# pragma omp for reduction ( + : total )
for ( i = 2; i <= n; i++ )
{
primo = 1;

for ( j = 2; j < i; j++ )
{
if ( i % j == 0 )
{
primo = 0;
break;
}
}
total = total + primo;
}

return total;
}