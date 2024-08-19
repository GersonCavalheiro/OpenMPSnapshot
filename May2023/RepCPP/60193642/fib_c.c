




















#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <nanos.h>

int cutoff_value = 10;

int fib_seq ( int n );
int fib_seq ( int n )
{
int x, y;

if ( n < 2 ) return n;

x = fib_seq( n-1 );

y = fib_seq( n-2 );

return x + y;
}

int fib ( int n, int d );

typedef struct {
int n;
int d;
int *x;
} fib_args;

void fib_1( void *ptr );
void fib_1( void *ptr )
{
fib_args * args = ( fib_args * )ptr;
*args->x = fib( args->n-1,args->d+1 );
}

void fib_2( void *ptr );
void fib_2( void *ptr )
{
fib_args * args = ( fib_args * )ptr;   
*args->x = fib( args->n-2,args->d+1 );
}

nanos_smp_args_t fib_device_arg_1 = { fib_1 };
nanos_smp_args_t fib_device_arg_2 = { fib_2 };



struct nanos_const_wd_definition_1
{
nanos_const_wd_definition_t base;
nanos_device_t devices[1];
};

struct nanos_const_wd_definition_1 const_data1 = 
{
{{
.mandatory_creation = true,
.tied = false},
__alignof__(fib_args),
0,
1,0,NULL},
{
{
nanos_smp_factory,
&fib_device_arg_1
}
}
};

struct nanos_const_wd_definition_1 const_data2 = 
{
{{
.mandatory_creation = true,
.tied = false},
__alignof__(fib_args),
0,
1,0,NULL},
{
{
nanos_smp_factory,
&fib_device_arg_2
}
}
};

nanos_wd_dyn_props_t dyn_props = {0};

int fib ( int n, int d )
{
int x, y;

if ( n < 2 ) return n;

if ( d < cutoff_value ) {
{
nanos_wd_t wd=0;
fib_args *args=0;

NANOS_SAFE( nanos_create_wd_compact ( &wd, &const_data1.base, &dyn_props, sizeof( fib_args ), ( void ** )&args,
nanos_current_wd(), NULL, NULL ) );
args->n = n;
args->d = d;
args->x = &x;

NANOS_SAFE( nanos_submit( wd,0,0,0 ) );
}

{
nanos_wd_t wd=0;
fib_args *args=0;

NANOS_SAFE( nanos_create_wd_compact ( &wd,  &const_data2.base, &dyn_props, sizeof( fib_args ), ( void ** )&args,
nanos_current_wd(), NULL, NULL ) );
args->n = n;
args->d = d;
args->x = &y;

NANOS_SAFE( nanos_submit( wd,0,0,0 ) );
}

NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
} else {
x = fib_seq( n-1 );
y = fib_seq( n-2 );
}

return x + y;
}

double get_wtime( void );
double get_wtime( void )
{

struct timeval ts;
double t;
int err;

err = gettimeofday( &ts, NULL );
t = ( double ) ( ts.tv_sec )  + ( double ) ts.tv_usec * 1.0e-6;

return t;
}

int fib0 ( int n );
int fib0 ( int n )
{
double start,end;
int par_res;

start = get_wtime();
par_res = fib( n,0 );
end = get_wtime();

printf( "Fibonacci result for %d is %d\n", n, par_res );
printf( "Computation time: %f seconds.\n",  end - start );
return par_res;
}


int main ( int argc, char **argv )
{
int n=25;

if ( argc > 1 ) n = atoi( argv[1] );

if ( fib0( n ) != 75025 ) return 1;

return 0;
}
