#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<sys/time.h>
#include<Kokkos_Core.hpp>
#include <cblas.h>
#if defined(KOKKOS_ENABLE_CUDA)
#include "CudaTimer.h"
#endif
#if defined( KOKKOS_ENABLE_HIP)
#include "HipTimer.h"
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
#include "OpenMPTimer.h"
#endif
#if defined(KOKKOS_ENABLE_SERIAL)
#include "SimpleTimer.h"
#endif
#if defined(KOKKOS_ENABLE_CUDA)
using Timer = CudaTimer;
#elif defined( KOKKOS_ENABLE_HIP)
using Timer = HipTimer;
#elif defined(KOKKOS_ENABLE_OPENMP)
using Timer = OpenMPTimer;
#else
using Timer = SimpleTimer;
#endif
int computeNbTeamsPerDot(int vector_length, int nbDots)
{
constexpr int workPerTeam = 4096;  
int teamsPerDot = 1;
int approxNumTeams =
(vector_length * nbDots) / workPerTeam;
if (approxNumTeams < 1)    approxNumTeams = 1;
if (approxNumTeams > 1024) approxNumTeams = 1024;
if (nbDots >= approxNumTeams) {
teamsPerDot = 1;
}
else {
teamsPerDot = approxNumTeams / nbDots;
}
return teamsPerDot;
}
void batched_dot_product(int nx, int ny, int nrepeat, bool use_lambda)
{
Kokkos::View<double**, Kokkos::LayoutLeft> x("X", nx, ny);
Kokkos::View<double**, Kokkos::LayoutLeft> y("Y", nx, ny);
Kokkos::View<double*>  dotProd("dot_prod", ny);
{
using md_policy = Kokkos::MDRangePolicy< Kokkos::Rank<2> >;
Kokkos::parallel_for(
"init_arrays",
md_policy({0,0},{nx,ny}),
KOKKOS_LAMBDA (const int& i, const int& j)
{
x(i,j) = 1.0/(i+1)*(j+1);
y(i,j) = 1.0/(i+1)*(j+1);
});
}
using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int>>;
using member_t = team_policy_t::member_type;
int nbTeams = ny;
const team_policy_t policy_lambda(nbTeams, Kokkos::AUTO(), Kokkos::AUTO());
auto dot_prod_lambda = KOKKOS_LAMBDA (const member_t& member)
{
double dot_prod = 0;
int j = member.league_rank();
Kokkos::parallel_reduce(
Kokkos::TeamThreadRange(member, nx),
[&](const int &i, double &update)
{
update += x(i,j) * y(i,j);
},
dot_prod);
Kokkos::single(Kokkos::PerTeam(member), [&]() { dotProd(j) = dot_prod; });
};
int nbTeamsPerDot = computeNbTeamsPerDot(nx,ny);
printf("Using nbTeamsPerDot = %d\n",nbTeamsPerDot);
int nbTeams2 = nbTeamsPerDot * ny;
printf("Using nbTeams = %d\n", nbTeams2);
const team_policy_t policy_lambda2(nbTeams2, Kokkos::AUTO());
auto dot_prod_lambda2 = KOKKOS_LAMBDA (const member_t& member)
{
double partial_dot_prod = 0;
int teamId = member.league_rank();
int j         = teamId / nbTeamsPerDot;
int pieceId   = teamId % nbTeamsPerDot;
int pieceSize = nx / nbTeamsPerDot;
int begin     =  pieceId      * pieceSize;
int end       = (pieceId + 1) * pieceSize;
if (pieceId == nbTeamsPerDot - 1) end = nx;
Kokkos::parallel_reduce(
Kokkos::TeamThreadRange(member, begin, end),
[&](const int &i, double &update)
{
update += x(i,j) * y(i,j);
},
partial_dot_prod);
Kokkos::single(Kokkos::PerTeam(member),
[&]() { Kokkos::atomic_add(&dotProd(j), partial_dot_prod); });
};
{
Timer timer;
timer.start();
for(int k = 0; k < nrepeat; k++)
{
Kokkos::parallel_for(
"compute_dot_products_lambda",
policy_lambda,
dot_prod_lambda);
}
timer.stop();
double time_seconds = timer.elapsed();
printf("Kokkos one team per dot:\n");
printf("#nx      ny        Time(s) TimePerIterations(s) size(MB) BW(GB/s)\n");
printf("%7i %7i   %8lf %20.3e  %3.3f %3.3f\n",
nx, ny,
time_seconds,
time_seconds/nrepeat,
(nx*ny*2+ny)*sizeof(double)*1.0e-6,
(nx*ny*2+ny)*sizeof(double)*nrepeat/time_seconds*1.0e-9);
}
{
Kokkos::deep_copy(dotProd, 0.0);
Timer timer;
timer.start();
for(int k = 0; k < nrepeat; k++)
{
Kokkos::parallel_for(
"compute_dot_products_lambda2",
policy_lambda2,
dot_prod_lambda2);
}
timer.stop();
double time_seconds = timer.elapsed();
printf("Kokkos %d team per dot:\n",nbTeamsPerDot);
printf("#nx      ny        Time(s) TimePerIterations(s) size(MB) BW(GB/s)\n");
printf("%7i %7i   %8lf %20.3e  %3.3f %3.3f\n",
nx, ny,
time_seconds,
time_seconds/nrepeat,
(nx*ny*2+ny)*sizeof(double)*1.0e-6,
(nx*ny*2+ny)*sizeof(double)*nrepeat/time_seconds*1.0e-9);
}
#ifdef USE_CBLAS_SERIAL
{
Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::OpenMP> x2("X2", nx, ny);
Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::OpenMP> y2("Y2", nx, ny);
Kokkos::View<double*, Kokkos::OpenMP>                      dotProd2("dot_prod2", ny);
{
using md_policy = Kokkos::MDRangePolicy< Kokkos::OpenMP, Kokkos::Rank<2> >;
Kokkos::parallel_for(
"init_arrays",
md_policy({0,0},{nx,ny}),
KOKKOS_LAMBDA (const int& i, const int& j)
{
x2(i,j) = 1.0/(i+1)*(j+1);
y2(i,j) = 1.0/(i+1)*(j+1);
});
}
OpenMPTimer timer;
timer.start();
for(int k = 0; k < nrepeat; k++)
{
#pragma omp parallel for
for (int j=0; j<ny; ++j)
{
const double *px = &x2(0,j);
const double *py = &y2(0,j);
double * pdot = dotProd2.data();
dotProd2[j] = cblas_ddot(nx, px, 1, py, 1);
}
} 
timer.stop();
double time_seconds = timer.elapsed();
printf("CBLAS serial:\n");
printf("#nx      ny        Time(s) TimePerIterations(s) size(MB) BW(GB/s)\n");
printf("%7i %7i   %8lf %20.3e  %3.3f %3.3f\n",
nx, ny,
time_seconds,
time_seconds/nrepeat,
(nx*ny*2+ny)*sizeof(double)*1.0e-6,
(nx*ny*2+ny)*sizeof(double)*nrepeat/time_seconds*1.0e-9);
}
#endif 
} 
int main(int argc, char* argv[]) {
int nx = 1000;          
int ny = 100;           
int nrepeat = 10;       
bool use_lambda = true; 
for(int i=0; i<argc; i++) {
if( strcmp(argv[i], "-nx") == 0) {
nx = atoi(argv[++i]);
} else if( strcmp(argv[i], "-ny") == 0) {
ny = atoi(argv[++i]);
} else if( strcmp(argv[i], "-nrepeat") == 0) {
nrepeat = atoi(argv[++i]);
} else if( strcmp(argv[i], "-use_lambda") == 0) {
int tmp = atoi(argv[++i]);
use_lambda = tmp!=0 ? true : false;
} else if( (strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0)) {
printf("Batched Dot Products options:\n");
printf("  -nx <int>:         length of column vectors (default: 1000)\n");
printf("  -ny <int>:         number of column vectors (default: 10)\n");
printf("  -nrepeat <int>:    number of integration invocations (default: 10)\n");
printf("  -use_lambda <int>: use lambda ? (default: 1)\n");
printf("  -help (-h):        print this message\n");
}
}
if (use_lambda)
printf("Using lambda  version\n");
else
printf("Using functor version\n");
Kokkos::initialize(argc,argv);
batched_dot_product(nx, ny, nrepeat, use_lambda);
Kokkos::finalize();
}
