# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <ctime>
# include <omp.h>
using namespace std;
# define NV 6
int main ( int argc, char **argv );
int *dijkstra_distance ( int ohd[NV][NV] );
void find_nearest ( int s, int e, int mind[NV], bool connected[NV], int *d, 
int *v );
void init ( int ohd[NV][NV] );
void timestamp ( void );
void update_mind ( int s, int e, int mv, bool connected[NV], int ohd[NV][NV], 
int mind[NV] );
int main ( int argc, char **argv )
{
int i;
int i4_huge = 2147483647;
int j;
int *mind;
int ohd[NV][NV];
timestamp ( );
cout << "\n";
cout << "DIJKSTRA_OPENMP\n";
cout << "  C++ version\n";
cout << "  Use Dijkstra's algorithm to determine the minimum\n";
cout << "  distance from node 0 to each node in a graph,\n";
cout << "  given the distances between each pair of nodes.\n";
cout << "\n";
cout << "  Although a very small example is considered, we\n";
cout << "  demonstrate the use of OpenMP directives for\n";
cout << "  parallel execution.\n";
init ( ohd );
cout << "\n";
cout << "  Distance matrix:\n";
cout << "\n";
for ( i = 0; i < NV; i++ )
{
for ( j = 0; j < NV; j++ )
{
if ( ohd[i][j] == i4_huge )
{
cout << "  Inf";
}
else
{
cout << "  " << setw(3) <<  ohd[i][j];
}
}
cout << "\n";
}
mind = dijkstra_distance ( ohd );
cout << "\n";
cout << "  Minimum distances from node 0:\n";
cout << "\n";
for ( i = 0; i < NV; i++ )
{
cout << "  " << setw(2) << i
<< "  " << setw(2) << mind[i] << "\n";
}
delete [] mind;
cout << "\n";
cout << "DIJKSTRA_OPENMP\n";
cout << "  Normal end of execution.\n";
cout << "\n";
timestamp ( );
return 0;
}
int *dijkstra_distance ( int ohd[NV][NV] )
{
bool *connected;
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
connected = new bool[NV];
connected[0] = true;
for ( i = 1; i < NV; i++ )
{
connected[i] = false;
}
mind = new int[NV];
for ( i = 0; i < NV; i++ )
{
mind[i] = ohd[0][i];
}
#pragma omp parallel private ( my_first, my_id, my_last, my_md, my_mv, my_step ) shared ( connected, md, mind, mv, nth, ohd )
{
my_id = omp_get_thread_num ( );
nth = omp_get_num_threads ( ); 
my_first =   (   my_id       * NV ) / nth;
my_last  =   ( ( my_id + 1 ) * NV ) / nth - 1;
#pragma omp single
{
cout << "\n";
cout << "  P" << my_id
<< ": Parallel region begins with " << nth << " threads.\n";
cout << "\n";
}
cout << "  P" << my_id
<< ":  First=" << my_first
<< "  Last=" << my_last << "\n";
for ( my_step = 1; my_step < NV; my_step++ )
{
#pragma omp single 
{
md = i4_huge;
mv = -1; 
}
find_nearest ( my_first, my_last, mind, connected, &my_md, &my_mv );
#pragma omp critical
{
if ( my_md < md )  
{
md = my_md;
mv = my_mv;
}
}
#pragma omp barrier
#pragma omp single 
{
if ( mv != - 1 )
{
connected[mv] = true;
cout << "  P" << my_id
<< ": Connecting node " << mv << "\n";;
}
}
#pragma omp barrier
if ( mv != -1 )
{
update_mind ( my_first, my_last, mv, connected, ohd, mind );
}
#pragma omp barrier
}
#pragma omp single
{
cout << "\n";
cout << "  P" << my_id
<< ": Exiting parallel region.\n";
}
}
delete [] connected;
return mind;
}
void find_nearest ( int s, int e, int mind[NV], bool connected[NV], int *d, 
int *v )
{
int i;
int i4_huge = 2147483647;
*d = i4_huge;
*v = -1;
for ( i = s; i <= e; i++ )
{
if ( !connected[i] && mind[i] < *d )
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
for ( i = 0; i < NV; i++ )  
{
for ( j = 0; j < NV; j++ )
{
if ( i == j )
{
ohd[i][i] = 0;
}
else
{
ohd[i][j] = i4_huge;
}
}
}
ohd[0][1] = ohd[1][0] = 40;
ohd[0][2] = ohd[2][0] = 15;
ohd[1][2] = ohd[2][1] = 20;
ohd[1][3] = ohd[3][1] = 10;
ohd[1][4] = ohd[4][1] = 25;
ohd[2][3] = ohd[3][2] = 100;
ohd[1][5] = ohd[5][1] = 6;
ohd[4][5] = ohd[5][4] = 8;
return;
}
void timestamp ( )
{
# define TIME_SIZE 40
static char time_buffer[TIME_SIZE];
const struct std::tm *tm_ptr;
std::time_t now;
now = std::time ( NULL );
tm_ptr = std::localtime ( &now );
std::strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr );
std::cout << time_buffer << "\n";
return;
# undef TIME_SIZE
}
void update_mind ( int s, int e, int mv, bool connected[NV], int ohd[NV][NV], 
int mind[NV] )
{
int i;
int i4_huge = 2147483647;
for ( i = s; i <= e; i++ )
{
if ( !connected[i] )
{
if ( ohd[mv][i] < i4_huge )
{
if ( mind[mv] + ohd[mv][i] < mind[i] )  
{
mind[i] = mind[mv] + ohd[mv][i];
}
}
}
}
return;
}
