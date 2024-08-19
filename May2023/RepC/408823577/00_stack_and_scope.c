#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#define _GNU_SOURCE
#ifndef _OPENMP
#error "openmp support is required to compile this code"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <omp.h>
int main( int argc, char **argv )
{
int     i;
int     nthreads = 1;
double *array_s, *array_p;
register unsigned long long base_of_stack asm("rbp");
register unsigned long long top_of_stack asm("rsp");
printf( "\nmain thread (pid: %d, tid: %ld) data:\n"
"base of stack is: %p\n"
"top of stack is : %p\n"
"&i is           : %p\n"
"   rbp - &i     : %td\n"
"   &i - rsp     : %td\n"
"\n\n",
getpid(), syscall(SYS_gettid),
(void*)base_of_stack,
(void*)top_of_stack,
&i,
(void*)base_of_stack - (void*)&i,
(void*)&i - (void*)top_of_stack );
array_s = (double*)malloc( 10*sizeof(double)); 
array_p = (double*)malloc( 10*sizeof(double));
printf( "main thread :\n"
"\tarray_s is at address %p\n"
"\tarray_p is at address %p\n",
array_s, array_p );
#pragma omp parallel
#pragma omp master
nthreads = omp_get_num_threads();
printf("using %d threads\n", nthreads);
size_t stack_base_addresses[ nthreads ];
size_t stack_top_addresses[ nthreads ];
#pragma omp parallel private(i, array_p)
{
int me = omp_get_thread_num();
unsigned long long my_stackbase;
unsigned long long my_stacktop;
__asm__("mov %%rbp,%0" : "=mr" (my_stackbase));
__asm__("mov %%rsp,%0" : "=mr" (my_stacktop));
stack_base_addresses[me] = my_stackbase;
stack_top_addresses[me] = my_stacktop;
printf( "thread (tid: %ld) nr %d:\n"
"\tmy base of stack is %p ( %td from main\'s stack )\n"
"\tmy i address is %p\n"
"\t\t%td from my stackbase and %td from main\'s\n",	    
syscall(SYS_gettid), me,
(void*)my_stackbase, (void*)base_of_stack - (void*)my_stackbase,
&i, (void*)&i - (void*)my_stackbase, (void*)&i - (void*)base_of_stack);
printf( "thread %d: I see array_s at address %p and array_p at address %p\n",
me, array_s, array_p );
}
printf( "\n" );
for( i = 0; i < nthreads; i++ )
{
printf("thread %d: bp @ %p, tp @ %p: %lld B", i,
(void*)stack_base_addresses[i],
(void*)stack_top_addresses[i],
(long long int)(stack_base_addresses[i]-stack_top_addresses[i]));
if (i > 0 )
printf("\t --> %lld B from top of thread %d",
(long long int)(stack_base_addresses[i]-stack_top_addresses[i-1]), i-1);
printf("\n");
}
printf("\n");
char *omp_stacksize = NULL;
if ( (omp_stacksize = getenv("OMP_STACKSIZE")) == NULL )
printf("you did not define OMP_STACKSIZE: try to set it and check what happens to the threads' stack pointers\n");
else
printf("you defined OMP_STACKSIZE as %s: try to change that value and check what happens to the threads' stack pointers\n",
omp_stacksize);
printf("\n");
free(array_p);
free(array_s);
return 0;
}
