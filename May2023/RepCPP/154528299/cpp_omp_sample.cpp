#include <iostream>
#include <omp.h> 
#include <array> 
#include <cmath>

using namespace std;





int main() {
int i, total, total2 ;
int nthreads, myid ;
const int Ndim = 10, chunk=4 ;
array<float,Ndim> arr ; 
float start, finish ;
int bthp ;
static int athp ;
omp_lock_t lck ; 
#pragma omp threadprivate (athp)
i=42 ; 
athp = 0 ;
bthp = 10 ;


cout << endl << "*** BASIC PARALLEL REGION ***" << endl ;
#pragma omp parallel
{
nthreads = omp_get_num_threads() ;
myid = omp_get_thread_num() ; 
#pragma omp critical 
{ 
cout << "Number of threads = " << nthreads << endl; 
cout <<  "Hello from thread number " << myid << endl ;
}
}
cout << endl ;


cout <<  "*** CRITICAL REGION ***"  << endl ;
total = 0 ;
#pragma omp parallel private(myid)
{
myid = omp_get_thread_num() ; 
#pragma omp critical 
{
total = total + myid ;
}
}
cout <<  "Total = " << total  << endl ;
cout << endl ;


cout <<  "*** PRIVATE VARIABLES ***" << endl ;
cout <<  "i before private region: " << i << endl ;
#pragma omp parallel default(shared) private(myid,i)
{
myid = omp_get_thread_num() ;
i = pow(myid,2) ;
#pragma omp critical 
{
cout <<  "Hello from thread number " << myid 
<< ". I have changed i to " << i << endl ;
}
}
cout <<  "i after private region: " << i << endl ;
cout << endl ;


cout <<  "*** PARALLEL FOR LOOP ***" << endl ;
#pragma omp parallel private(i,myid)
{
myid = omp_get_thread_num() ;
#pragma omp barrier

#pragma omp for nowait
for (i=0;i<10;i++){

#pragma omp critical
{
cout << "Hello from thread number " << myid << ". I am doing iteration " 
<< i << endl ;
}

}

#pragma omp critical 
{
cout << "My ID is " << myid << ", I didn't wait for my friends to finish." 
<< endl ;
}
}

cout << endl ;


cout <<  "*** ATOMIC REGION ***"  << endl ;
total = 0 ;
#pragma omp parallel private(myid)
{
myid = omp_get_thread_num() ;
#pragma omp atomic
total = total + myid ;
}
cout <<  "Total = " << total << endl ;
cout << endl ;


cout <<  "*** REDUCTION REGION ***"  << endl ;
total = 0 ;
total2 = 0 ;
#pragma omp parallel reduction(+:total, total2) private(myid)
{
myid = omp_get_thread_num() ;
total = total + myid ;
total2 = total2 + 2*myid ;
}
cout <<  "Total = " << total << ", Total2 = " << total2  << endl ;
cout << endl ;


cout <<  "*** SECTIONS ***"  << endl ;
#pragma omp parallel private(myid)
{
myid = omp_get_thread_num() ;
#pragma omp sections
{		
#pragma omp section
#pragma omp critical 
{
cout <<  "Thread " << myid << " thinks it's Duck season."  << endl ;
}
#pragma omp section
#pragma omp critical 
{
cout <<  "Thread " << myid << " thinks it's Rabbit season."  << endl ;
}
#pragma omp section
#pragma omp critical 
{
cout << "Thread " << myid << " doesn't care what season it is." << endl;
}
}
}
cout << endl ;


cout <<  "*** SINGLE / COPYPRIVATE***"  << endl ;
#pragma omp parallel private(myid,i)
{
myid = omp_get_thread_num() ;

i = 2*myid ;
#pragma omp critical 
{
cout <<  "My ID is " << myid << " and i= " << i  << endl ;
}

#pragma omp barrier 

#pragma omp single copyprivate(i)
{		
i=20 ;
cout << "My ID is " << myid << ", I arrived first and set i=" << i << endl ;
}

#pragma omp critical 
{
cout << "My ID is " << myid << " and i=" << i << endl ;
}
}
cout << endl ;


cout <<  "*** ORDERED ***"  << endl ;
#pragma omp parallel private(myid,i)
{
myid = omp_get_thread_num() ;

#pragma omp for ordered 
for (i=0 ; i<Ndim ; i++){

arr[i] = 2.0*float(i*myid) ; 

#pragma omp ordered
{
cout << "My ID is " << myid << ", I'm working on iteration " << i 
<< " and arr[i]=" << arr[i] << endl ;
}

arr[i] = arr[i]/float(myid) ; 
}
}
cout << endl ;


cout <<  "*** THREAD PRIVATE / COPYIN ***"  << endl ;
cout << "About to enter the parallel region, athp=" << athp  << endl ;
cout << endl ;

#pragma omp parallel private(myid,i)
{
myid = omp_get_thread_num() ;
athp = 2*myid + 1 ;
#pragma omp critical 
{
cout << "In the 1st parallel region, my ID is " << myid 
<< ", my athp =" << athp << endl ;
}
}

cout << endl ;
cout <<  "Just left the first parallel region, athp=" << athp << endl ;
cout << endl ;

#pragma omp parallel private(myid,i)
{
myid = omp_get_thread_num() ;
#pragma omp critical 
{
cout <<  "In the 2nd parallel region, my ID is" << myid 
<< ", my athp=" << athp << endl ;
}
}

cout << endl ;
cout <<  "Just left the 2nd parallel region, athp=" << athp << endl ;
cout << endl ;

#pragma omp parallel copyin(athp) private(myid)
{
myid = omp_get_thread_num() ;
#pragma omp critical 
{
cout <<  "In the 3rd parallel region, my ID is" << myid 
<< ", my athp=" << athp << endl ;
}
}
cout << endl ;


cout <<  "*** FIRST PRIVATE ***"  << endl ;
cout <<  "About to enter the parallel region, bthp=" << bthp  << endl ;
cout << endl ;
#pragma omp parallel private(myid) firstprivate(bthp)
{
myid = omp_get_thread_num() ;
bthp = bthp + myid ;
#pragma omp critical 
{
cout <<  "In the parallel region, my ID is " << myid 
<< ", my bthp =" << bthp << endl ;
}
}
cout << endl ;
cout <<  "Just left the parallel region, bthp=" << bthp << endl ;
cout << endl ;


cout <<  "*** LAST PRIVATE ***"  << endl ;
cout <<  "About to enter the parallel region, bthp=" << bthp << endl ;
cout << endl ;

#pragma omp parallel for private(i) lastprivate(bthp)
for (i=1 ; i<9 ; i++) {
myid = omp_get_thread_num() ;
bthp = i ;
#pragma omp critical 
{
cout << "My ID is " << myid << ", I'm working on iteration " << i
<< "and bthp=" << bthp  << endl ;
}
}

cout << endl ;
cout <<  "Just left the parallel region, bthp =" << bthp << endl ;
cout << endl ;


cout <<  "*** IF ***"  << endl ;
i=10 ;
#pragma omp parallel if(i==10) private(myid)
{
myid = omp_get_thread_num() ;
#pragma omp critical 
{
cout <<  "My ID is " << myid 
<< ", executing in parallel, so i must equal 10." << endl ;
}
}
cout << endl ;
i=11 ;
#pragma omp parallel if(i==10) private(myid)
{
myid = omp_get_thread_num() ;
#pragma omp critical 
{
cout << "My ID is " << myid << ", I'm alone so i must not be 10." << endl ;
}
}
cout << endl ;


cout <<  "*** NUM THREADS ***"  << endl ;
#pragma omp parallel num_threads(2)
{
nthreads = omp_get_num_threads() ;
#pragma omp critical 
{
cout <<  "For this parallel region only the number of threads is "
<< nthreads << endl ;
}
}
cout << endl ;


cout <<  "*** SCHEDULING ***"  << endl ;

#pragma omp parallel num_threads(4) firstprivate(bthp) private(myid, i)
{
myid = omp_get_thread_num() ;
#pragma omp for schedule(dynamic, 1)
for (i=0; i<17 ; i++) {
if (i>13) {
bthp = pow(sin(float(i)),10) ;
} else {
bthp = i ;
}
#pragma omp critical 
{
cout << "My ID is "<< myid <<", I'm working on iteration " << i << endl; 
}
}
}
cout << endl ;


cout <<  "*** SIMPLE LOCK ***" << endl ;
omp_init_lock(&lck) ;
#pragma omp parallel shared(lck) private(myid, i)
{
myid = omp_get_thread_num() ;
i = 0 ;


omp_set_lock(&lck) ;
#pragma omp critical 
{
cout << "My ID is " << myid 
<< ". I own the lock and am about to release it." << endl ;
}
omp_unset_lock(&lck) ;
#pragma omp critical 
{
cout << "My ID is " << myid << ". I just released the lock." << endl ;
}

while (not omp_test_lock(&lck)) {
i++ ;
}

#pragma omp critical 
{
cout << "My ID is " << myid << ", I counted to " << i 
<< " while waiting for the lock." << endl ;
}
omp_unset_lock(&lck) ;

}
omp_destroy_lock(&lck);

return 0 ;
}


