#include "stdio.h"
#include <omp.h>
struct s{
float value;
} Array[4];
int main( int argc, char *argv[ ] )
{
omp_set_num_threads(NUMTHREADS);
int someBigNumber = 1000000000;
double time0 = omp_get_wtime( );
#pragma omp parallel for
for( int i = 0; i < 4; i++ ){
float privateVar = Array[i].value; 
for( int j = 0; j < someBigNumber; j++ ){
privateVar = privateVar + 2.;
}
Array[i].value = privateVar;
}
double time1 = omp_get_wtime( );
double timeElapsed = time1- time0;
double megaAddsPerSecond = (double)someBigNumber*(double)4/timeElapsed/1000000.;
FILE *fp;
fp = fopen("fix2.txt", "a");
fprintf (fp, "%d\t", NUMPAD);
fprintf (fp, "%d\t", NUMTHREADS);
fprintf (fp, "%f\t",megaAddsPerSecond);
fprintf (fp, "%f\t",1000000.*timeElapsed);
fprintf (fp, "Fix 2");
fprintf (fp, "\n");
printf("Threads: %d\n", NUMTHREADS);
printf("Padding: %d\n", NUMPAD);
printf("Elapsed Time=%10.2lf microseconds\n",1000000.*timeElapsed);
printf("Average Performance = %8.2lf MegaAdds/Sec\n", megaAddsPerSecond);
printf ("Fix 1\n");
printf("***** \n");
return 0;
}