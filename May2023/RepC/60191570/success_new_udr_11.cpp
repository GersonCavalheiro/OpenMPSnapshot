#include <stdlib.h>
#include "omp.h"
typedef struct {
int x ;
int y ;
} point_t;
int max(int x, int y ) 
{
( x ) > ( y ) ? ( x ) : ( y ) ;
}
#pragma omp declare reduction( maxarea : point_t : omp_out.x = max(omp_out.x , omp_in.x ), omp_out.y = max(omp_out.y , omp_in.y ) ) initializer( omp_priv = {0 ,0} )
int main (int argc, char* argv[])
{
point_t pt;
#pragma omp parallel reduction (maxarea : pt)
pt;
return 0;
}
