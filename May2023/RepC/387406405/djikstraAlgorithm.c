#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#define NV 800
int main ( int argc, char** argv );
int *dijkstra_distance ( int ohd[NV][NV] );
void find_nearest ( int s, int e, int mind[NV], int connected[NV], int *d, int *v );
void init ( int ohd[NV][NV] );
void timestamp ( void);
void update_mind ( int s, int e, int mv, int connected[NV], int ohd[NV][NV], int mind[NV] );
int main ( int argc, char **argv )
{
int i;
int i4_huge = 2147483647;
int j;
int *mind;
int ohd[NV][NV];
omp_set_num_threads(5);
init ( ohd );
clock_tbegin = clock();
mind = dijkstra_distance ( ohd );
clock_tend = clock();
printf("Execution Time : %lf\n\n", (double)(end -begin) / CLOCKS_PER_SEC);
free ( mind );
fprintf ( stdout, "\n");
return 0;
}
int *dijkstra_distance ( intohd[NV][NV] )
{
int *connected;
int i;
int i4_huge = 2147483647;
int md;
int *mind;
int mv;
int my_first;
int my_id;
int my_last;
int my_md;
int my_mv;
int my_step;
int nth;
connected = ( int* ) malloc ( NV * sizeof( int) );
connected[0] = 1;
for( i = 1; i < NV; i++ )
{
connected[i] = 0;
}
mind = (int* )malloc ( NV * sizeof( int) );
for( i = 0; i < NV; i++ )
{
mind[i] = ohd[0][i];
}
#pragma omp parallel private ( my_first, my_id, my_last, my_md, my_mv, my_step ) shared ( connected, md, mind, mv, nth, ohd )
{
my_id = omp_get_thread_num ( );
nth = omp_get_num_threads ( );
my_first = ( my_id * NV ) / nth;
my_last = ( ( my_id + 1) * NV ) / nth -1;
for( my_step = 1; my_step < NV; my_step++ )
{
#pragma omp single
{
md = i4_huge;
mv = -1;
}
find_nearest ( my_first, my_last, mind, connected, &my_md, &my_mv );
#pragma omp critical
{
if( my_md < md )
{
md = my_md;
mv = my_mv;
}
}
#pragma omp barrier# pragma omp single
{
if( mv != -1)
{
connected[mv] = 1;
}
}
#pragma omp barrierif( mv != -1)
{
update_mind ( my_first, my_last, mv, connected, ohd, mind );
}
#pragma omp barrier
}
}
free ( connected );
return mind;
}
void find_nearest ( int s, int e, int mind[NV], int connected[NV], int *d, int *v )
{
int i;
int i4_huge = 2147483647;
*d = i4_huge;
*v = -1;
for( i = s; i <= e; i++ )
{
if( !connected[i] && ( mind[i] < *d ) )
{
*d = mind[i];
*v = i;
}
}
return;
}
void init ( int ohd[NV][NV] )
{
int i;
int i4_huge = 2147483647;
int j;
for( i = 0; i < NV; i++ )
{
for( j = 0; j < NV; j++ )
{
if( i == j )
{
ohd[i][i] = 0;
}
else
{
ohd[i][j] = rand() % 1000;
}
}
}
return;
}
void update_mind ( int s, int e, int mv, int connected[NV], int ohd[NV][NV], int mind[NV] )
{
int i;
int i4_huge = 2147483647;
for( i = s; i <= e; i++ )
{
if( !connected[i] )
{
if( ohd[mv][i] < i4_huge )
{
if( mind[mv] + ohd[mv][i] < mind[i] )
{
mind[i] = mind[mv] + ohd[mv][i];
}
}
}
}
return;
}
