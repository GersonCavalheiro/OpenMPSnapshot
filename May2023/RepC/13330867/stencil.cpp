#include <par-res-kern_general.h>
#include <Grappa.hpp>
#include <FullEmpty.hpp>
using namespace Grappa;
#define DOUBLE
#ifdef DOUBLE
#define DTYPE     double
#define MPI_DTYPE MPI_DOUBLE
#define EPSILON   1.e-8
#define COEFX     1.0
#define COEFY     1.0
#define FSTR      "%lf"
#else
#define DTYPE     float
#define MPI_DTYPE MPI_FLOAT
#define EPSILON   0.0001f
#define COEFX     1.0f
#define COEFY     1.0f
#define FSTR      "%f"
#endif
#define root 0
#define symmetric static
#define INDEXIN(i,j)  (i+RADIUS+(j+RADIUS)*(width+2*RADIUS))
#define IN(i,j)       in[INDEXIN(i-istart,j-jstart)]
#define INDEXOUT(i,j) (i+(j)*(width))
#define OUT(i,j)      out[INDEXOUT(i-istart,j-jstart)]
#define WEIGHT(ii,jj) weight[ii+RADIUS][jj+RADIUS]
int main(int argc, char * argv[]) {
Grappa::init( &argc, &argv );
symmetric int my_ID = Grappa::mycore();
if (my_ID == root) {
std::cout<<"Parallel Research Kernels version "<<PRKVERSION<<std::endl;
std::cout<<"Grappa stencil execution on 2D grid"<<std::endl;
}
#ifndef STAR
if (my_ID == root)
std::cout <<"ERROR: Compact stencil not supported"<<std::endl;
exit(1);     
#endif
if (argc != 3){
if (my_ID == root)
std::cout<<"Usage:"<<argv[0]<<" <# iterations> <array dimension>"<<std::endl;
exit(1);
}
int iterations  = atoi(argv[1]);
if (iterations < 1){
if (my_ID == root)
std::cout<<"ERROR: iterations must be >= 1 :"<<iterations<<std::endl;
exit(1);
}
int n        = atoi(argv[2]);
long nsquare = (long) n * n;
if (nsquare < Grappa::cores()){
if (my_ID == root)
std::cout<<"ERROR: grid size "<<nsquare<<" must be at least # cores "<<
Grappa::cores()<<std::endl;
exit(1);
}
Grappa::run([iterations,n]{
int Num_procsx, Num_procsy;
int Num_procs=Grappa::cores();
if (RADIUS < 0) {
std::cout<<"ERROR: Stencil radius "<<RADIUS<<" should be non-negative"<<std::endl;
exit(1);
}
if (2*RADIUS +1 > n) {
std::cout<<"ERROR: Stencil radius "<<RADIUS<<" exceeds grid size "<<n<<std::endl;
exit(1);
}
factor(Num_procs, &Num_procsx, &Num_procsy);
symmetric int my_IDx;
symmetric int my_IDy;
on_all_cores( [Num_procsx] {
my_IDx = my_ID%Num_procsx;
my_IDy = my_ID/Num_procsx; }
);
std::cout<<"Number of cores        = "<<Num_procs<<std::endl;
std::cout<<"Grid size              = "<<n<<std::endl;
std::cout<<"Radius of stencil      = "<<RADIUS<<std::endl;
std::cout<<"Tiles in x/y-direction = "<<Num_procsx<<"/"<<Num_procsy<<std::endl;
std::cout<<"Type of stencil        = star"<<std::endl;
#ifdef DOUBLE
std::cout<<"Data type              = double precision"<<std::endl;
#else
std::cout<<"Data type              = single precision"<<std::endl;
#endif
#if LOOPGEN
std::cout<<"Script used to expand stencil loop body"<<std::endl;
#else
std::cout<<"Compact representation of stencil loop body"<<std::endl;
#endif
std::cout<<"Number of iterations   = "<<iterations<<std::endl;
symmetric double start;
symmetric double total;
symmetric long istart;
symmetric long iend;
symmetric long jstart;
symmetric long jend;
symmetric long width;
symmetric long height;
symmetric FullEmpty<DTYPE> * left_halo;
symmetric FullEmpty<DTYPE> * right_halo;
symmetric FullEmpty<DTYPE> * top_halo;
symmetric FullEmpty<DTYPE> * bottom_halo;
symmetric FullEmpty<bool> * CTS_top;
symmetric FullEmpty<bool> * CTS_bottom;
symmetric FullEmpty<bool> * CTS_right;
symmetric FullEmpty<bool> * CTS_left;
symmetric DTYPE * in;
symmetric DTYPE * out;
symmetric DTYPE weight[2*RADIUS+1][2*RADIUS+1];
Grappa::on_all_cores( [n,Num_procs,Num_procsx,Num_procsy]{
width = n/Num_procsx;
int leftover = n%Num_procsx;
if (my_IDx<leftover) {
istart = (width+1) * my_IDx;
iend = istart + width;
}
else {
istart = (width+1) * leftover + width * (my_IDx-leftover);
iend = istart + width - 1;
}
width = iend - istart + 1;
if (width == 0) {
std::cout<<"ERROR: core "<<my_ID<<" has no work to do"<<std::endl;
exit(1);
}
height = n/Num_procsy;
leftover = n%Num_procsy;
if (my_IDy<leftover) {
jstart = (height+1) * my_IDy;
jend = jstart + height;
}
else {
jstart = (height+1) * leftover + height * (my_IDy-leftover);
jend = jstart + height - 1;
}
height = jend - jstart + 1;
if (height == 0) {
printf("ERROR: core %d has no work to do\n", my_ID);
exit(1);
}
if (width < RADIUS || height < RADIUS) {
std::cout<<"ERROR: core "<<my_ID<<" has no work to do"<<std::endl;
exit(1);
}
long total_length_in = (width+2*RADIUS)*(height+2*RADIUS);
long total_length_out = width*height;
in  = Grappa::locale_new_array<DTYPE>(total_length_in);
out = Grappa::locale_new_array<DTYPE>(total_length_out);
if (!in || !out) {
std::cout<<"ERROR: core "<<my_ID<<
" could not allocate space for input/output array"<<std::endl;
exit(1);
}
for (int jj=-RADIUS; jj<=RADIUS; jj++) for (int ii=-RADIUS; ii<=RADIUS; ii++)
WEIGHT(ii,jj) = (DTYPE) 0.0;
for (int ii=1; ii<=RADIUS; ii++) {
WEIGHT(0, ii) = WEIGHT( ii,0) =  (DTYPE) (1.0/(2.0*ii*RADIUS));
WEIGHT(0,-ii) = WEIGHT(-ii,0) = -(DTYPE) (1.0/(2.0*ii*RADIUS));
}
for (int j=jstart; j<=jend; j++) for (int i=istart; i<=iend; i++) {
IN(i,j)  = COEFX*i+COEFY*j;
OUT(i,j) = (DTYPE)0.0;
}
top_halo    = Grappa::locale_new_array<Grappa::FullEmpty<DTYPE>>(RADIUS*width);
bottom_halo = Grappa::locale_new_array<Grappa::FullEmpty<DTYPE>>(RADIUS*width);
right_halo  = Grappa::locale_new_array<Grappa::FullEmpty<DTYPE>>(RADIUS*height);
left_halo   = Grappa::locale_new_array<Grappa::FullEmpty<DTYPE>>(RADIUS*height);
if (!top_halo || !bottom_halo || !right_halo || !left_halo) {
std::cout<<"ERROR: Rank "<<my_ID<<" could not allocate communication buffers"<<std::endl;
exit(1);
}
for (int i=0; i<RADIUS*width; i++) {
top_halo[i].reset();
bottom_halo[i].reset();
}
for (int i=0; i<RADIUS*height; i++) {
right_halo[i].reset();
left_halo[i].reset();
}
CTS_top     = new FullEmpty<bool>();
CTS_bottom  = new FullEmpty<bool>();
CTS_right   = new FullEmpty<bool>();
CTS_left    = new FullEmpty<bool>();
writeXF( CTS_top, true);
writeXF( CTS_bottom, true);
writeXF( CTS_right, true);
writeXF( CTS_left, true);
} );
Grappa::finish( [n,Num_procsx,Num_procsy, iterations] {
Grappa::on_all_cores( [n,Num_procsx,Num_procsy, iterations] {
int right_nbr  = my_ID+1;
int left_nbr   = my_ID-1;
int top_nbr    = my_ID+Num_procsx;
int bottom_nbr = my_ID-Num_procsx;
for (int iter = 0; iter<=iterations; iter++){
int i, j, ii, jj, kk;
if (iter == 1) start = Grappa::walltime();
if (my_IDy < Num_procsy-1 && readFE( CTS_top))	    
for (kk=0,j=jend-RADIUS+1; j<=jend; j++)
for (i=istart; i<=iend; i++,kk++) {
auto val = IN(i,j);
Grappa::delegate::call<async>( top_nbr, [=] () {
writeXF( &bottom_halo[kk], val);
} );
}
if (my_IDy > 0 && readFE( CTS_bottom))
for (kk=0,j=jstart; j<=jstart+RADIUS-1; j++)
for (i=istart; i<=iend; i++,kk++) {
auto val = IN(i,j);
Grappa::delegate::call<async>( bottom_nbr, [=] () {
writeXF( &top_halo[kk], val);
} );
}
if (my_IDx < Num_procsx-1 && readFE( CTS_right))
for (kk=0,j=jstart; j<=jend; j++)
for (i=iend-RADIUS+1; i<=iend; i++,kk++) {
auto val = IN(i,j);
Grappa::delegate::call<async>( right_nbr, [=] () {
writeXF( &left_halo[kk], val);
} );
}
if (my_IDx > 0 && readFE( CTS_left))
for (kk=0,j=jstart; j<=jend; j++)
for (i=istart; i<=istart+RADIUS-1; i++,kk++) {
auto val = IN(i,j);
Grappa::delegate::call<async>( left_nbr, [=] () {
writeXF( &right_halo[kk], val);
} );
}
if (my_IDy < Num_procsy-1) {
for (kk=0,j=jend+1; j<=jend+RADIUS; j++)
for (i=istart; i<=iend; i++,kk++) {
IN(i,j) = readFE( &top_halo[kk]);
}
Grappa::delegate::call<async>( top_nbr, [=] () {
writeXF( CTS_bottom, true);
} );
}
if (my_IDy > 0) {
for (kk=0,j=jstart-RADIUS; j<=jstart-1; j++)
for (i=istart; i<=iend; i++,kk++) {
IN(i,j) = readFE( &bottom_halo[kk]);
}
Grappa::delegate::call<async>( bottom_nbr, [=] () {
writeXF( CTS_top, true);
} );
}
if (my_IDx < Num_procsx-1) {
for (kk=0,j=jstart; j<=jend; j++)
for (i=iend+1; i<=iend+RADIUS; i++,kk++)
IN(i,j) = readFE( &right_halo[kk]);
Grappa::delegate::call<async>( right_nbr, [=] () {
writeXF( CTS_left, true);
} );
}
if (my_IDx > 0) {
for (kk=0,j=jstart; j<=jend; j++) 
for (i=istart-RADIUS; i<=istart-1; i++,kk++)
IN(i,j) = readFE( &left_halo[kk]);
Grappa::delegate::call<async>( left_nbr, [=] () {
writeXF( CTS_right, true);
} );
}
for (j=MAX(jstart,RADIUS); j<=MIN(n-RADIUS-1,jend); j++) {
for (i=MAX(istart,RADIUS); i<=MIN(n-RADIUS-1,iend); i++) {
#if LOOPGEN
#include "loop_body_star.incl"
#else
for (jj=-RADIUS; jj<=RADIUS; jj++) OUT(i,j) += WEIGHT(0,jj)*IN(i,j+jj);
for (ii=-RADIUS; ii<0; ii++)       OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
for (ii=1; ii<=RADIUS; ii++)       OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
#endif
}
}
for (j=jstart; j<=jend; j++) for (i=istart; i<=iend; i++) IN(i,j)+= 1.0;
} 
} );
} );       
symmetric DTYPE local_norm;
Grappa::on_all_cores ( [n] {
int my_ID=Grappa::mycore();
total = Grappa::walltime() - start;
local_norm = (DTYPE) 0.0;
for (int j=MAX(jstart,RADIUS); j<=MIN(n-RADIUS-1,jend); j++) {
for (int i=MAX(istart,RADIUS); i<=MIN(n-RADIUS-1,iend); i++) {
local_norm += (DTYPE)ABS(OUT(i,j));
}
}
});
double reference_norm = (DTYPE) (iterations+1) * (COEFX + COEFY);
double actual_norm = Grappa::reduce<DTYPE,collective_sum<DTYPE>>(&local_norm );
double f_active_points = (DTYPE) (n-2*RADIUS)*(DTYPE) (n-2*RADIUS);
actual_norm /= f_active_points;
if (ABS(reference_norm-actual_norm) >= EPSILON) {
std::cout<<"ERROR: checksum "<<actual_norm<<
"  does not match verification value "<<reference_norm<<std::endl;
}
else {
double iter_time = Grappa::reduce<double,collective_max<double>>( &total );
int stencil_size = 4*RADIUS+1;
double flops = (DTYPE) (2*stencil_size+1) * f_active_points;
double avgtime = iter_time/(double)iterations;
std::cout << "Solution validates"<<std::endl;
std::cout << "Rate (MFlops/s): " << 1.0E-06*flops/avgtime<<
"  Avg time (s): "<<avgtime<<std::endl;
}
});
Grappa::finalize();
return 0;
}
