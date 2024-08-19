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

#define COMPUTE_MFLOPS(fp_ops, max_cyc, PAPI_mhz)         (PAPI_mhz>0 && max_cyc/PAPI_mhz>0) ?   fp_ops/(max_cyc/PAPI_mhz) : 0

bool verbose_mode=false;

extern "C" void dummy(const char* format, ...);

#include "adjoint_stacks.c"

extern long adjoint_stack_size;

ulong **stack_sizes_overview=NULL;
map< int , ulong** > mem_statistics;

void 
before_reverse_mode()
{ 
if(stack_sizes_overview!=NULL) {

stack_sizes_overview[omp_get_thread_num()][0] = a1_STACKc_c;
stack_sizes_overview[omp_get_thread_num()][1] = a1_STACKi_c;
stack_sizes_overview[omp_get_thread_num()][2] = a1_STACKf_c;
}
}

void print_memory_statistics_details() 
{
cerr << "Adjoint mode stack overview: " << endl;
for(map< int , ulong** >::const_iterator it=mem_statistics.begin(); it!=mem_statistics.end(); it++ ) {
int   num_threads = it->first;
ulong** table = it->second;
cerr << "Test with " << num_threads << " threads:\n";
cerr << "Thread_ID |          STACKc         |          STACKi          |        STACKf    " << endl;
cerr << "----------------------------------------------------------------------------------\n";
float overall_mb=0;
for(int _i=0;_i<num_threads;_i++) {
uint STACKc_num_elements = table[_i][0]; 
float STACKc_mb = STACKc_num_elements*sizeof(int)/(1024.f*1024.f);
uint STACKi_num_elements = table[_i][1];
float STACKi_mb = STACKi_num_elements*sizeof(int)/(1024.f*1024.f);
uint STACKf_num_elements = table[_i][0];
float STACKf_mb = STACKf_num_elements*sizeof(double)/(1024.f*1024.f);
overall_mb+=STACKc_mb+STACKi_mb+STACKf_mb;
cerr << setw(6) << _i << "    |    " << setw(8) << STACKc_num_elements << " (" << fixed << setprecision(2) << STACKc_mb << ") mb" << setw(3) << " "
<< "|" << setw(8) << STACKi_num_elements << " (" << fixed <<setprecision(2) << STACKi_mb << ") mb" <<setw(8) << " "
<< "|" << setw(8) << STACKf_num_elements << " (" << fixed << setprecision(2) << STACKf_mb << ") mb"
<< endl;
}
cerr << "------------------------------------------------------------------------------------------------\n";
cerr << "Adjoint stacks used " << overall_mb << " mb with " << num_threads << " threads.\n";
}
}

int main(int argc, char* argv[])
{
#include "test_specific_decl.c"
long mem_free_kb=0;
double elapsed_time=0.;

int m;
double start,end;
long n = 1e8;
long nfourth;
long i=0;
long j=0;

if(argc!=2) { cerr << "error: The syntax is " << argv[0] << " <number of threads>\n"; return 1; }

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

int omp_num_threads=atoi(argv[1]);
while( (n%omp_num_threads)!=0 ) n--;
omp_set_num_threads(omp_num_threads);
assert( (n%omp_num_threads)==0 );
int p = omp_num_threads ;

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
DEFINE_ARR( t1_thread_result, omp_get_num_procs() );
DEFINE_ARR( a1_thread_result, omp_get_num_procs() );

cerr << "Init data" << endl;
ifstream ifs("/proc/meminfo"); assert(ifs.good());
while( ifs.good() ) {
string line;
const string searchpattern = "MemFree:  ";
getline(ifs, line) ;
if( line.find(searchpattern)!=string::npos )  {
cout << line << endl;
line.erase( 0, string(searchpattern).length()-1 );
line.erase( line.length()-2 );
istringstream iss(line);
iss >> mem_free_kb;
cout << "There is " << mem_free_kb/1024. << "mb memory free." << endl;
n = oneMB_double;
while( (n*n+n)*2*sizeof(double)/1024 > (mem_free_kb*25/100) ) 
n--;
while( (n%omp_get_num_procs())!=0 )
n--;

adjoint_stack_size_kb=(mem_free_kb*90/100) - ((n+n*n)*2*sizeof(double)/1024);

break;
}
}
ifs.close();
cerr << "info: n has size "  << n << ".\n";
cerr << "info: x vector has "  << (unsigned)n/1000 << " * 1e3.\n";
cerr << "info: adjoint stack size (in mb): "  << (ulong)adjoint_stack_size_kb/1024 << ".\n";
DEFINE_ARR(x, n);
DEFINE_ARR(t1_x, n);
double *a1_x = t1_x;

DEFINE_MTX(A, n);
DEFINE_MTX(t1_A, n);
double *a1_A = t1_A;

srand( time(NULL) );

#ifdef  PAPI
map<int, long long> threads_max_cyc;
map<int, int>       threads_MFLOPS;

map<int, long long> t1_threads_max_cyc;
map<int, int>       t1_threads_MFLOPS;

map<int, long long> a1_threads_max_cyc;
map<int, int>       a1_threads_MFLOPS;
#endif

#pragma omp parallel for private(i)
for(i=0;i<n;i++) {
x[i]=fabs(sin((double)rand()));
t1_x[i] = a1_x[i]=0.; 
}
#pragma omp parallel for private(i,j)
for(i=0;i<n;i++) {
for(j=0;j<n;j++) {
A[i*n+j]=fabs(sin((double)rand()));
t1_A[i*n+j] = a1_A[i*n+j] = 0.; 
}
}

map<int, double>	threads_elapsed_time;
map<int, double>	t1_threads_elapsed_time;
map<int, double>	a1_threads_elapsed_time;

cerr << "Adjoint scale test with " << omp_get_max_threads()<< " threads.\n";
p = omp_get_max_threads();
stack_sizes_overview = new ulong*[omp_num_threads]; assert(stack_sizes_overview);
for(i=0;i<omp_num_threads;i++) { stack_sizes_overview[i] = new ulong[3]; assert(stack_sizes_overview[i]); }
for(int _i=0;_i<omp_num_threads;_i++) for(int _j=0;_j<3;_j++) stack_sizes_overview[_i][_j] = 0;

cerr << "Start parallel region with " << omp_get_max_threads() << " thread(s) ... \n";
#pragma omp parallel
{ a1_STACKc_init(); a1_STACKi_init(); a1_STACKf_init(); }
for(int k=0;k<n;k++) t1_x[k]=0.; 
for(int k=0;k<n;k++) for(int l=0;l<n;l++) a1_A[k*n+l]=0.; 
a1_thread_result[0]=1.;
#ifdef  PAPI
#include "test_specific_init.c"
#include "a1_test.in.papi.withCstack"

a1_threads_max_cyc[p] = max_cyc;
a1_threads_MFLOPS[p]  = COMPUTE_MFLOPS(fp_ops,max_cyc,PAPI_mhz);
#else
#include "test_specific_init.c"
#include "a1_test.in.spl.withCstack"
#endif
a1_threads_elapsed_time[p] = elapsed_time;
#pragma omp parallel
{ a1_STACKc_deallocate(); a1_STACKi_deallocate(); a1_STACKf_deallocate(); }
mem_statistics[omp_num_threads] = stack_sizes_overview;
stack_sizes_overview = NULL;



cerr << "Write results for scaling test to plot files ... \n";
string cmd_argv (argv[0]);
unsigned found = cmd_argv.rfind('/');
if (found!=std::string::npos) 
cmd_argv = cmd_argv.substr(found+1);
ostringstream oss; oss << "." << cmd_argv << "." << time(NULL) << ".";
string filename_a(hostname + oss.str() + "adjoint_speedup.plot");
string filename_mem(hostname + oss.str() + "adjoint_stack_mem.plot");
cerr << "Try to open file " << filename_a << " ... ";
ofstream ofs_adjoint(filename_a.c_str());   assert(ofs_adjoint.good());
cerr <<"done.\n";
ostringstream ossa;

ossa.precision(2);
ossa << fixed;
#ifdef PAPI
string headline_for_plotfile("# Threads   Speedup      MFLOPS     Timespeedup");
ossa << headline_for_plotfile << endl;
cerr << "DEBUG: " << a1_threads_elapsed_time.size() << endl;
for(map<int,double>::const_iterator it=a1_threads_elapsed_time.begin();it!=a1_threads_elapsed_time.end();it++) {
int num_threads = it->first; double elapsed_time=it->second;
ossa << setw(5) << " " << num_threads << setw(6) << " " << setw(6)<< a1_threads_max_cyc[1]/(double)a1_threads_max_cyc[num_threads] 
<< setw(5) << " " << setw(6)<< a1_threads_MFLOPS[num_threads] 
<< setw(8) << " " << setw(6) << a1_threads_elapsed_time[1]/(double)a1_threads_elapsed_time[num_threads] << endl;
}
#else
string headline_for_plotfile("# Threads   Speedup ");
ossa << headline_for_plotfile << endl;
for(map<int,double>::const_iterator it=a1_threads_elapsed_time.begin();it!=a1_threads_elapsed_time.end();it++) {
int num_threads = it->first; double elapsed_time=it->second;
ossa << setw(5) << " " << num_threads << setw(6) << " " << setw(6)<< a1_threads_elapsed_time[1]/(double)a1_threads_elapsed_time[num_threads]  << endl;
}
#endif

cerr << "\tData from adjoint code.\n";
cerr << ossa.str() << endl;    ofs_adjoint << ossa.str() << endl;
cerr << "\tFile " << filename_a << " done \n\n"; ofs_adjoint.close();
cerr << "done\n";

omp_destroy_lock(&lock_var); 
return 0;
}

