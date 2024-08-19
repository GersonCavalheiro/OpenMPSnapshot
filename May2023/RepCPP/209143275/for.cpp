#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <assert.h>
#include <unistd.h>
#include <map>
#ifdef PAPI
#include <papi.h>
#endif

#include "cstack_wrapper.h"
#include "cstack.h"

using namespace std;

#include "utilities.c"
uint *a1_STACKc = NULL;   ulong a1_STACKc_c = 0;
int *a1_STACKi = NULL;    ulong a1_STACKi_c = 0;
double *a1_STACKf = NULL; ulong a1_STACKf_c = 0;
#pragma omp threadprivate (a1_STACKc, a1_STACKc_c, a1_STACKi, a1_STACKi_c, a1_STACKf, a1_STACKf_c)
uint *t2_STACKc = NULL;   ulong t2_STACKc_c = 0;
int *t2_STACKi = NULL;    ulong t2_STACKi_c = 0;
double *t2_STACKf = NULL; ulong t2_STACKf_c = 0;
#pragma omp threadprivate (t2_STACKc, t2_STACKc_c, t2_STACKi, t2_STACKi_c, t2_STACKf, t2_STACKf_c)

int main(int argc, char* argv[])
{
#include "test_specific_decl.c"
long mem_free_kb=0;
int p = omp_get_max_threads() ;
double elapsed_time=0.;

int m;
double start,end;
long n = 1e8;
long i=0;
long j=0;


char buffer[1024];
size_t pos=string(argv[0]).find_last_of('/'); 
if(pos==string::npos) pos=0;
string filename("/tmp/" + string(argv[0]).substr(pos) + ".tmp");
string cmd = "hostname > " + filename;
assert( !system(cmd.c_str()) );
FILE *fp = fopen(filename.c_str(), "r"); assert(fp);
assert( fgets(buffer, 1024, fp)==buffer );
fclose(fp);
cmd = "rm " + filename;  assert( !system(cmd.c_str()) );
string hostname(buffer);   
hostname = hostname.substr(0,hostname.length()-1);
assert(hostname.length()>0);

#ifdef PAPI
const int number_of_events = 2;
int* EventSets = new int [p];
long long **PAPI_values = new long long* [p];
long long fp_ops=0;
long long mflops=0;
long long max_cyc=0;
long long PAPI_elapsed_cyc;
int PAPI_mhz;

for(i=0;i<p;i++) { EventSets[i] = PAPI_NULL; }
int global_PAPI_retval = PAPI_library_init( PAPI_VER_CURRENT ); assert(global_PAPI_retval == PAPI_VER_CURRENT );
assert( omp_get_nested()==0 ); 
global_PAPI_retval = PAPI_thread_init( ( unsigned long (*)( void ) ) ( omp_get_thread_num ) ); assert(global_PAPI_retval==PAPI_OK);
#endif

omp_lock_t lock_var;
omp_init_lock(&lock_var);

DEFINE_ARR( thread_result, omp_get_num_procs() );
DEFINE_ARR( t2_thread_result, omp_get_num_procs() );
DEFINE_ARR( a1_thread_result, omp_get_num_procs() );
DEFINE_ARR( t2_a1_thread_result, omp_get_num_procs() );

ifstream ifs("/proc/meminfo"); assert(ifs.good());
while( ifs.good() ) {
string line;
const string searchpattern = "MemTotal:  ";
getline(ifs, line) ;
if( line.find(searchpattern)!=string::npos )  {
cout << line << endl;
line.erase( 0, string(searchpattern).length()-1 );
line.erase( line.length()-2 );
istringstream iss(line);
iss >> mem_free_kb;
cout << "There is " << mem_free_kb/1024. << "mb memory free." << endl;
n = oneMB_double;
while( (n*n+n)*2*sizeof(double)/1024 > (mem_free_kb*15/100) ) 
n--;

if( mem_free_kb/(1024*1024) > 48 ) {
while( (n*n+n)*2*sizeof(double)/(1024*1024*1024) < 256 )
n++;
}
while( (n%omp_get_num_procs())!=0 )
n--;

if( (mem_free_kb*90/100) > ((n+n*n)*2*sizeof(double)/1024) )
adjoint_stack_size_kb=(mem_free_kb*90/100) - ((n+n*n)*2*sizeof(double)/1024);
else
adjoint_stack_size_kb=(mem_free_kb*90/100);
break;
}
}
ifs.close();
n=100000;
cerr << "info: n has size "  << n << ".\n";
cerr << "info: x vector has "  << (unsigned)n/1000 << " * 1e3.\n";
cerr << "info: A matrix has "  << (unsigned long long)(n*n)/1000 << " * 1e3.\n";
cerr << "info: adjoint stack size (in mb): "  << (long)adjoint_stack_size_kb/1024 << ".\n";
cerr << "Test (n%omp_get_num_procs())==0 successful.\n";
DEFINE_ARR(x, n);
DEFINE_ARR(t2_x, n);
DEFINE_ARR(a1_x, n);
DEFINE_ARR(t2_a1_x, n);

DEFINE_MTX(A, n);
DEFINE_MTX(t2_A, n);
DEFINE_MTX(a1_A, n);
DEFINE_MTX(t2_a1_A, n);

srand( time(NULL) );

#ifdef  PAPI
long double ratio_real_to_measuread_FP_ops=1.;
ulong 			    overall_fp_ops=0;
map<int, long long> threads_max_cyc;
map<int, int>       threads_MFLOPS;
#endif
cerr << "Start of scale test with "<< omp_get_num_threads() << " threads ...\n";
#pragma omp parallel for private(i)
for(i=0;i<n;i++) {
x[i]=fabs(sin((double)rand()));
a1_x[i] = t2_x[i] = t2_a1_x[i]=0.; 
}
#pragma omp parallel for private(i,j)
for(i=0;i<n;i++) { for(j=0;j<n;j++) {
A[i*n+j]=fabs(sin((double)rand()));
a1_A[i*n+j] = t2_A[i*n+j] = t2_a1_A[i*n+j]=0.; 
}
}
#pragma omp parallel
{ a1_STACKc_init(); a1_STACKi_init(); a1_STACKf_init(); }
#pragma omp parallel
{ t2_STACKc_init(); t2_STACKi_init(); t2_STACKf_init(); }
cerr << "forward over reverse test\n";
#include "t2_a1_test.in.spl.spl.withCstack"

omp_destroy_lock(&lock_var); 
return 0;
}

