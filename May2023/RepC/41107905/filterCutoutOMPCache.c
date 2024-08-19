#include<stdio.h>
#include<stdint.h>
#include<omp.h>
#include<stdbool.h>
#define ALIGNMENT 4096
void sortDataFirst (uint32_t * cutoutarray, int cutoutsize,  uint32_t * filterlist, int listsize )
{
for ( int i=0; i<listsize; i++)
{
printf("HELLO");
}
}
void filterCutoutOMPCache( uint32_t * cutout, int cutoutsize, uint32_t * filterlist, int listsize)
{
int i,j;
bool equal;
posix_memalign ( (void**)&cutoutarray, ALIGNMENT, cutoutsize * sizeof(uint32_t) );
memcpy ( cutoutarray, cutout, cutoutsize * sizeof(uint32_t) );
for ( int filter_len = 16; filter_len<=listsize; filter_len*=2 )
printf("MAX THREADS: %d",omp_get_max_threads());
#pragma omp parallel num_threads(omp_get_max_threads()) 
{
#pragma omp for private(i,j,equal) schedule(dynamic)
for ( i=0; i<cutoutsize; i++)
{
equal = false;
for( j=0; j<listsize; j++)
{
if( cutout[i] == filterlist[j] )
{
equal = true;
break;
}
}
if( !equal || cutout[i] > filterlist[j] )
cutout[i] = 0;
}
int ID = omp_get_thread_num();
printf("THREAD ID: %d",ID);
}
}
