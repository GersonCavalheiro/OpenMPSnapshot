#include "stdio.h"
#include <omp.h>
struct s{
float value;
int pad[4];
}Array[4];
int main( int argc, char *argv[ ] )
{
const  int  SomeBigNumber = 100000000;  
omp_set_num_threads(NUMTHREADS);
double time0 = omp_get_wtime( );
#pragma omp parallel for
for( int i = 0; i < 4; i++ ){
for( int j = 0; j < SomeBigNumber; j++ ){
Array[ i ].value  =  Array[ i ].value + 2.;
}
}
double time1 = omp_get_wtime( );
double timeElapsed = time1- time0;
double megaAddsPerSecond = (double)SomeBigNumber*(double)4/timeElapsed/1000000.;
FILE *fp;
fp = fopen("fix1.txt", "a");
fprintf (fp, "%d\t", NUMPAD);
fprintf (fp, "%d\t", NUMTHREADS);
fprintf (fp, "%f\t",megaAddsPerSecond);
fprintf (fp, "%f\t",1000000.*timeElapsed);
fprintf (fp, "Fix 1");
fprintf (fp, "\n");
printf("Threads: %d\n", NUMTHREADS);
printf("Padding: %d\n", NUMPAD);
printf("Average Performance = %8.2lf MegaAdds/Sec\n", megaAddsPerSecond);
printf("Elapsed Time=%10.2lf microseconds\n",1000000.*timeElapsed);
printf ("Fix 1\n");
printf("***** \n");
return 0;
}
