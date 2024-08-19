


















#include<stdio.h>
#include"nanos.h"



int main ( int argc, char *argv[] )
{
int error = 0, a = 0;

#pragma omp task shared(a) inout(a)
{
a++;
fprintf(stderr,"1");
nanos_dependence_release_all();
nanos_yield();
usleep(10000);
fprintf(stderr,"3");
a--;
}

#pragma omp task shared(a,error) in(a)
{
fprintf(stderr,"2");
if (!a) error++;
}

#pragma omp taskwait

fprintf(stderr,"4:verification=%s\n",error?"UNSUCCESSFUL":"successful");

return error;
}
