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
#define UPPER_BOUNDARY_FOR_ERROR_PERCENTAGE       33.

#define COMPUTE_MFLOPS(fp_ops, max_cyc, PAPI_mhz)         (PAPI_mhz>0 && max_cyc/PAPI_mhz>0) ?   fp_ops/(max_cyc/PAPI_mhz) : 0
#define ADJUST_FP_OPS(overall_fp_ops)  \
ratio_real_to_measuread_FP_ops=1.;   \
if(num_threads==1) overall_fp_ops=fp_ops*ratio_real_to_measuread_FP_ops; \
else if(num_threads==2){  \
cerr << "last fp_ops before scaling: " << fp_ops<< endl; \
fp_ops=fp_ops*ratio_real_to_measuread_FP_ops; \
cerr << "last fp_ops after scaling : " << fp_ops<< endl; \
cerr << "overall_fp_ops            : " << overall_fp_ops<< endl; \
const float diff_in_percent = (float)abs(overall_fp_ops-fp_ops)*100/fp_ops ; \
if( diff_in_percent<UPPER_BOUNDARY_FOR_ERROR_PERCENTAGE/2. ) {   \
cerr << GREEN << "diff in %: " << diff_in_percent << NORMAL << endl; \
} \
else  \
cerr << RED << "diff in %: " << diff_in_percent << NORMAL << endl; \
cerr << "\n\n"; \
assert( diff_in_percent<UPPER_BOUNDARY_FOR_ERROR_PERCENTAGE );   \
}

bool verbose_mode=true;

extern "C" void dummy(const char* format, ...);

#include "adjoint_stacks.c"


ulong **stack_sizes_overview=NULL;
map< ulong , ulong** > mem_statistics;

void 
before_reverse_mode()
{ 
if(stack_sizes_overview!=NULL) {

stack_sizes_overview[omp_get_thread_num()][0] = a1_STACKc_c;
stack_sizes_overview[omp_get_thread_num()][1] = a1_STACKi_c;
stack_sizes_overview[omp_get_thread_num()][2] = a1_STACKf_c;
}
}

void print_memory_statistics(string filename) 
{
ofstream ofs(filename.c_str());
cerr << "Adjoint mode stack overview: " << endl;
float last_overall_mb=0;
ofs << "# Thread_ID         STACKc (in Mb)         STACKi  (in Mb)       STACKf  (in Mb)  " << endl;
for(map< ulong, ulong** >::const_iterator it=mem_statistics.begin(); it!=mem_statistics.end(); it++ ) {
int   num_threads = it->first;
ulong** table = it->second;
float overall_mb=0;
float STACKc_max=0;
float STACKi_max=0;
float STACKf_max=0;
for(int _i=0;_i<num_threads;_i++) {
ulong STACKc_num_elements = table[_i][0]; 
float STACKc_mb = STACKc_num_elements*sizeof(uint)/(1024.f*1024.f);
ulong STACKi_num_elements = table[_i][1];
float STACKi_mb = STACKi_num_elements*sizeof(int)/(1024.f*1024.f);
ulong STACKf_num_elements = table[_i][0];
float STACKf_mb = STACKf_num_elements*sizeof(double)/(1024.f*1024.f);
overall_mb+=STACKc_mb+STACKi_mb+STACKf_mb;
if( STACKc_mb>STACKc_max ) STACKc_max=STACKc_mb;
if( STACKf_mb>STACKf_max ) STACKf_max=STACKf_mb;
if( STACKi_mb>STACKi_max ) STACKi_max=STACKi_mb;
}
ofs << setw(6) << num_threads << setw(15) << " " 
<< setw(8) << fixed << setprecision(2) << STACKc_max << setw(15) << " "
<< setw(8) << fixed << setprecision(2) << STACKi_max << setw(15) << " "
<< setw(8) << fixed << setprecision(2) << STACKf_max << setw(15) << " "
<< endl;
if( last_overall_mb && fabs(last_overall_mb-overall_mb)>1 ) {
cerr << "warning: Adjoint stacks differs: previous " << last_overall_mb << "   current=" << overall_mb<< endl;
}
last_overall_mb=overall_mb;
}
cerr << " done." << endl;
}


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
DEFINE_ARR( t1_thread_result, omp_get_num_procs() );
DEFINE_ARR( a1_thread_result, omp_get_num_procs() );

cerr << "Init data" << endl;
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
cerr << "info: n has size "  << n << ".\n";
cerr << "info: x vector has "  << (unsigned)n/1000 << " * 1e3.\n";
cerr << "info: A matrix has "  << (unsigned long long)(n*n)/1000 << " * 1e3.\n";
cerr << "info: adjoint stack size (in mb): "  << (long)adjoint_stack_size_kb/1024 << ".\n";
cerr << "Test (n%omp_get_num_procs())==0 successful.\n";
DEFINE_ARR(x, n);
DEFINE_ARR(t1_x, n);
double *a1_x = t1_x;

DEFINE_MTX(A, n);
DEFINE_MTX(t1_A, n);
double *a1_A = t1_A;

srand( time(NULL) );

cerr << "First, we test if there is the correct derivative value d thread_result[0] / d x[0]\n";
long nbackup = n;
n = 100;
while( (n%omp_get_num_procs())!=0 ) n++;
cerr << "Init data ... ";
#pragma omp parallel for private(i)
for(i=0;i<n;i++) {
x[i]=fabs(sin((double)rand()));
t1_x[i] = a1_x[i] = 0.; 
}
#pragma omp parallel for private(i,j)
for(i=0;i<n;i++) {
for(j=0;j<n;j++) {
A[i*n+j]=fabs(sin((double)rand()));
t1_A[i*n+j] = a1_A[i*n+j] = 0.; 
}
}
cerr << "done";
i=0;
const long double epsilon = 1e-1;
#include "finite_differences_kernel.c"
n = nbackup;
cerr << "Comparison between finite differences and tangent-linear as well as the adjoint code was \e[1;32m  successful  \e[0;0m.\n";


#ifdef  PAPI
long double ratio_real_to_measuread_FP_ops=1.;
ulong 			    overall_fp_ops=0;
ulong 			    t1_overall_fp_ops=0;
ulong 			    a1_overall_fp_ops=0;
map<int, long long> threads_max_cyc;
map<int, int>       threads_MFLOPS;

map<int, long long> t1_threads_max_cyc;
map<int, int>       t1_threads_MFLOPS;

map<int, long long> a1_threads_max_cyc;
map<int, int>       a1_threads_MFLOPS;

char* PAPISRC;
PAPISRC = getenv("PAPISRC");
assert( PAPISRC!=NULL );
string s;
s = "cd " + string(PAPISRC) + "; make test 2>&1 1>/tmp/papitest;";
system( s.c_str() );
ifstream tmp("/tmp/papitest"); 
string line;
while(tmp.good()) {
getline(tmp, line);
if(line=="Test type    : 	           1") {
getline(tmp, line);
string pattern("PAPI_FP_INS  : ");
if(line.substr(0, pattern.length())==pattern) {
istringstream iss_fp_ops(line.substr(pattern.length())); 
long long measured_fp_ops;
const long long real_fp_ops = 40000000 ;
iss_fp_ops >> measured_fp_ops;
ratio_real_to_measuread_FP_ops = (long double)real_fp_ops/measured_fp_ops;
}
}
}
tmp.close();
cerr << "info: Ratio between measured and real FP_OPS is "  << ratio_real_to_measuread_FP_ops << ".\n";
#endif
map<int, double>	threads_elapsed_time;
map<int, double>	t1_threads_elapsed_time;
map<int, double>	a1_threads_elapsed_time;

cerr << "Start of scale test with up to "<< omp_get_num_procs() << " cores ...\n";
for(int num_threads=1;num_threads<=omp_get_num_procs();num_threads*=2) {
assert( (n%omp_get_num_procs())==0 );
cerr << "Scaling test with " << num_threads << " threads.\n";
omp_set_num_threads(num_threads);
p=num_threads;
#pragma omp parallel for private(i)
for(i=0;i<n;i++) {
x[i]=fabs(sin((double)rand()));
t1_x[i] = a1_x[i]=0.; 
}
#pragma omp parallel for private(i,j)
for(i=0;i<n;i++) { for(j=0;j<n;j++) {
A[i*n+j]=fabs(sin((double)rand()));
t1_A[i*n+j] = a1_A[i*n+j]=0.; 
}
}

#ifndef ONLY_ADJOINT
cerr << "Original code\n";
cerr << "Start parallel region with " << p << " thread(s) ... \n";
#ifdef  PAPI
#include "test_specific_init.c"
#include "test.in.papi.withCstack"

ADJUST_FP_OPS(overall_fp_ops)
threads_max_cyc[num_threads] = max_cyc;
threads_MFLOPS[num_threads]  = COMPUTE_MFLOPS(fp_ops,max_cyc,PAPI_mhz);
#else
#include "test_specific_init.c"
#include "test.in.spl.withCstack"
#endif
threads_elapsed_time[num_threads] = elapsed_time;

cerr << "tangent-linear test\n";
cerr << "Start parallel region with " << p << " thread(s) ... \n";
for(int k=0;k<n;k++) t1_x[k]=0.; 
for(int k=0;k<n;k++) for(int l=0;l<n;l++) t1_A[k*n+l]=0.; 
t1_x[0]=1.;
#ifdef  PAPI
#include "test_specific_init.c"
#include "t1_test.in.papi.withCstack"

ADJUST_FP_OPS(t1_overall_fp_ops)
t1_threads_max_cyc[num_threads] = max_cyc;
t1_threads_MFLOPS[num_threads]  = COMPUTE_MFLOPS(fp_ops,max_cyc,PAPI_mhz);
#else
#include "test_specific_init.c"
#include "t1_test.in.spl.withCstack"
#endif
t1_threads_elapsed_time[num_threads] = elapsed_time;
#endif

cerr << "Adjoint scale test\n";
stack_sizes_overview = new ulong*[num_threads]; assert(stack_sizes_overview);
for(i=0;i<num_threads;i++) { stack_sizes_overview[i] = new ulong[3]; assert(stack_sizes_overview[i]); }
for(int _i=0;_i<num_threads;_i++) for(int _j=0;_j<3;_j++) stack_sizes_overview[_i][_j] = 0;
for(int k=0;k<n;k++) t1_x[k]=0.; 
for(int k=0;k<n;k++) for(int l=0;l<n;l++) a1_A[k*n+l]=0.; 

cerr << "Start parallel region with " << p << " thread(s) ... \n";
#pragma omp parallel
{ a1_STACKc_init(); a1_STACKi_init(); a1_STACKf_init(); }
a1_thread_result[0]=1.;
#ifdef  PAPI
#include "test_specific_init.c"
#include "a1_test.in.papi.withCstack"

ADJUST_FP_OPS(a1_overall_fp_ops)
a1_threads_max_cyc[num_threads] = max_cyc;
a1_threads_MFLOPS[num_threads]  = COMPUTE_MFLOPS(fp_ops,max_cyc,PAPI_mhz);
#else
#include "test_specific_init.c"
#include "a1_test.in.spl.withCstack"
#endif
a1_threads_elapsed_time[num_threads] = elapsed_time;
#pragma omp parallel
{ a1_STACKc_deallocate(); a1_STACKi_deallocate(); a1_STACKf_deallocate(); }
mem_statistics[num_threads] = stack_sizes_overview;
stack_sizes_overview = NULL;
}

cerr << "Write results for scaling test to plot files ... \n";
string cmd_argv (argv[0]);
unsigned found = cmd_argv.rfind('/');
if (found!=std::string::npos) 
cmd_argv = cmd_argv.substr(found+1);
ostringstream oss; oss << "." << cmd_argv << "." << time(NULL) << ".";
#ifdef PAPI
string headline_for_plotfile("# Threads   Speedup      MFLOPS     Timespeedup");
#else
string headline_for_plotfile("# Threads   Speedup ");
#endif

#ifndef ONLY_ADJOINT
string filename_o(hostname + oss.str() + "original_speedup.plot");
string filename_t(hostname + oss.str() + "tangent_speedup.plot");
cerr << "Try to open file " << filename_o << " ... ";
ofstream ofs_original(filename_o.c_str());  assert(ofs_original.good());
cerr <<"done.\n";
cerr << "Try to open file " << filename_t << " ... ";
ofstream ofs_tangent(filename_t.c_str());   assert(ofs_tangent.good());
cerr <<"done.\n";
cerr << "\tData from original code.\n";
ostringstream osso;
ostringstream osst;
osso.precision(2); osst.precision(2); 
osso << fixed; osst << fixed; 
#ifdef PAPI
osso << headline_for_plotfile << endl;
osst << headline_for_plotfile << endl;
for(int num_threads=1;num_threads<=omp_get_max_threads();num_threads*=2) {
osso << setw(5) << " " << num_threads << setw(6) << " " << setw(6) << threads_max_cyc[1]/(double)threads_max_cyc[num_threads]       
<< setw(5) << " " << setw(6)<<    threads_MFLOPS[num_threads] 
<< setw(8) << " " << setw(6) << threads_elapsed_time[1]/(double)threads_elapsed_time[num_threads] << endl;
osst << setw(5) << " " << num_threads << setw(6) << " " << setw(6)<< t1_threads_max_cyc[1]/(double)t1_threads_max_cyc[num_threads] 
<< setw(5) << " " << setw(6)<< t1_threads_MFLOPS[num_threads] 
<< setw(8) << " " << setw(6) << t1_threads_elapsed_time[1]/(double)t1_threads_elapsed_time[num_threads] << endl;
}
#else
osso << headline_for_plotfile << endl;
osst << headline_for_plotfile << endl;
for(map<int,double>::const_iterator it=threads_elapsed_time.begin();it!=threads_elapsed_time.end();it++) {
int num_threads = it->first; double elapsed_time=it->second;
osso << setw(5) << " " << num_threads << setw(6) << " " << setw(6) << threads_elapsed_time[1]/(double)threads_elapsed_time[num_threads] << endl;
osst << setw(5) << " " << num_threads << setw(6) << " " << setw(6)<< t1_threads_elapsed_time[1]/(double)t1_threads_elapsed_time[num_threads]  << endl;
}
#endif
cerr << osso.str() << endl;    ofs_original << osso.str() << endl;
cerr << "\tFile " << filename_o << " done \n\n"; ofs_original.close(); 

cerr << "\tData from tangent-linear code.\n";
cerr << osst.str() << endl;    ofs_tangent << osst.str() << endl;
cerr << "\tFile " << filename_t << " done \n\n"; ofs_tangent.close(); 
#endif

ostringstream ossa;
string filename_a(hostname + oss.str() + "adjoint_speedup.plot");
string filename_mem(hostname + oss.str() + "adjoint_stack_mem.plot");
cerr << "Try to open file " << filename_a << " ... ";
ofstream ofs_adjoint(filename_a.c_str());   assert(ofs_adjoint.good());
cerr <<"done.\n";
ossa.precision(2);
ossa << fixed;

#ifdef PAPI
ossa << headline_for_plotfile << endl;
for(int num_threads=1;num_threads<=omp_get_max_threads();num_threads*=2) {
ossa << setw(5) << " " << num_threads << setw(6) << " " << setw(6)<< a1_threads_max_cyc[1]/(double)a1_threads_max_cyc[num_threads] 
<< setw(5) << " " << setw(6)<< a1_threads_MFLOPS[num_threads] 
<< setw(8) << " " << setw(6) << a1_threads_elapsed_time[1]/(double)a1_threads_elapsed_time[num_threads] << endl;
}
#else
ossa << headline_for_plotfile << endl;
for(map<int,double>::const_iterator it=threads_elapsed_time.begin();it!=threads_elapsed_time.end();it++) {
int num_threads = it->first; double elapsed_time=it->second;
ossa << setw(5) << " " << num_threads << setw(6) << " " << setw(6)<< a1_threads_elapsed_time[1]/(double)a1_threads_elapsed_time[num_threads]  << endl;
}
#endif
cerr << "\tData from adjoint code.\n";
cerr << ossa.str() << endl;    ofs_adjoint << ossa.str() << endl;
cerr << "\tFile " << filename_a << " done \n\n"; ofs_adjoint.close();
cerr << "done\n";
cerr << "Memory statistics:\n";  print_memory_statistics(filename_mem);



omp_destroy_lock(&lock_var); 
return 0;
}

