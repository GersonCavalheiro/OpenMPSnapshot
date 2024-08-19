{
assert( (n%omp_get_num_procs())==0 );
const long double h       = 1e-6;
const int indep_x_index = i;
const int dep_thread_result_index = i;
if(verbose_mode) cerr << "TEST: indep x["<<indep_x_index<<"] and dep thread_result["<<dep_thread_result_index<<"].\n";

srand( time(NULL) );
if(verbose_mode) cerr << "Init data ... ";
#pragma omp parallel for private(i)
for(i=0;i<n;i++) {
x[i]=fabs( (sin((double)rand())+1)*( (rand()%10)+1 ) );
t1_x[i] = a1_x[i]=0.; 
}
#pragma omp parallel for private(i,j)
for(i=0;i<n;i++) {
for(j=0;j<n;j++) {
A[i*n+j]=fabs( (sin((double)rand())+1)*( (rand()%10)+1 ) );
t1_A[i*n+j] = a1_A[i*n+j] = 0.; 
}
}
if(verbose_mode) cerr << "done.\n";


cerr.precision(10);
for(i=0;i<omp_get_num_procs();i++) { a1_thread_result[i]=t1_thread_result[i]=thread_result[i]=0.; }
#include "test_specific_init.c"
long double x0   = x[indep_x_index];
long double x0ph = x[indep_x_index] + 0.5*h;
long double x0mh = x[indep_x_index] - 0.5*h;
if(verbose_mode) { 
cerr << " x0  :" << x0 << endl; 
cerr << " x0+h:" << x0ph << endl; 
cerr << " x0-h:" << x0mh << endl;
}
x[indep_x_index]=x0ph;
#include "test.in.spl.withCstack"
long double fx0ph = thread_result[dep_thread_result_index];

for(i=0;i<omp_get_num_procs();i++) { a1_thread_result[i]=t1_thread_result[i]=thread_result[i]=0.; }
#include "test_specific_init.c"
x[indep_x_index]=x0mh;
#include "test.in.spl.withCstack"
long double fx0mh = thread_result[dep_thread_result_index];
if(verbose_mode) { cerr << " f(x+h): " << fx0ph << endl; cerr << " f(x-h): " << fx0mh << endl; }
long double deriv = (fx0ph - fx0mh)/h;

x[indep_x_index]=x0;
for(i=0;i<omp_get_num_procs();i++) { a1_thread_result[i]=t1_thread_result[i]=thread_result[i]=0.; }
for(int k=0;k<n;k++) t1_x[k]=0.; 
for(int k=0;k<n;k++) for(int l=0;l<n;l++) t1_A[k*n+l]=0.; 
t1_x[indep_x_index]=1.;
#include "test_specific_init.c"
#include "t1_test.in.spl.withCstack"
if(verbose_mode) { cerr << scientific << deriv <<" - " << t1_thread_result[dep_thread_result_index] << " = " << deriv-t1_thread_result[dep_thread_result_index]<< endl; }
assert( fabs(deriv - t1_thread_result[dep_thread_result_index]) < epsilon );

for(i=0;i<n;i++) { a1_x[i] = 0.; }
for(i=0;i<omp_get_num_procs();i++) { a1_thread_result[i]=thread_result[i]=0.; }
for(int k=0;k<n;k++) a1_x[k]=0.; 
for(int k=0;k<n;k++) for(int l=0;l<n;l++) a1_A[k*n+l]=0.; 

#pragma omp parallel
{ a1_STACKc_init(); a1_STACKi_init(); a1_STACKf_init(); }
a1_thread_result[dep_thread_result_index]=1.;
thread_result[0]=1.2345678; 
#include "test_specific_init.c"
#include "a1_test.in.spl.withCstack"
if(thread_result[0]!=1.2345678) cerr << endl << thread_result[0] <<endl;
assert( thread_result[0]==1.2345678 ); 
if(verbose_mode) cerr << "adjoint test successfull.\n";
#pragma omp parallel
{ a1_STACKc_deallocate(); a1_STACKi_deallocate(); a1_STACKf_deallocate(); }

#if 0
assert(thread_access_set);
for(int k=0;k<omp_get_num_procs();k++) {
for(int l=0;l<omp_get_num_procs();l++) {
if(l==k) continue;
for(set<void*>::const_iterator it=thread_access_set[k].begin(); it!=thread_access_set[k].end(); it++) {
for(set<void*>::const_iterator it2=thread_access_set[l].begin(); it2!=thread_access_set[l].end(); it2++) {
assert(*it!=*it2);
}
}
}
}
#endif

if(verbose_mode) { cerr << scientific << deriv<<" - " << a1_x[indep_x_index] << " = " << deriv-a1_x[indep_x_index]<< endl; }
assert( fabs(deriv - a1_x[indep_x_index]) < epsilon );
}
