



#include <CL/sycl.hpp>
#include "Timings.h"
#include "RCBForceTree.h"
#include "Partition.h"

#include <cstring>
#include <cstdio>
#include <ctime>
#include <stdexcept>
#include <assert.h>
using namespace std;

#ifdef SYCL_LANGUAGE_VERSION
#include "syclUtil.h"
#include <cmath>
#include <chrono>
#include <algorithm>

#define TILEX 4     
#define TILEY 4     
#define BLOCKX 32   
#define BLOCKY 4    
#define MAXX 32     
#define MAXY 256    

#define ALIGNX(n) ((n+TILEX-1)/TILEX*TILEX)  
#define ALIGNY(n) ((n+TILEY-1)/TILEY*TILEY)  

syclDeviceSelector __selector__;
#endif

namespace {
template <int TDPTS>
struct sphdesign {};

#define DECLARE_SPHDESIGN(TDPTS) \
template <> \
struct sphdesign<TDPTS> \
{ \
static const POSVEL_T x[TDPTS]; \
static const POSVEL_T y[TDPTS]; \
static const POSVEL_T z[TDPTS]; \
}; \


DECLARE_SPHDESIGN(1)
DECLARE_SPHDESIGN(2)
DECLARE_SPHDESIGN(3)
DECLARE_SPHDESIGN(4)
DECLARE_SPHDESIGN(6)
DECLARE_SPHDESIGN(12)
DECLARE_SPHDESIGN(14)

#undef DECLARE_SPHDESIGN


const POSVEL_T sphdesign<1>::x[] = {
0
};

const POSVEL_T sphdesign<1>::y[] = {
0
};

const POSVEL_T sphdesign<1>::z[] = {
0
};

const POSVEL_T sphdesign<2>::x[] = {
1.0,
-1.0
};

const POSVEL_T sphdesign<2>::y[] = {
0,
0
};

const POSVEL_T sphdesign<2>::z[] = {
0,
0
};

const POSVEL_T sphdesign<3>::x[] = {
1.0,
-.5,
-.5
};

const POSVEL_T sphdesign<3>::y[] = {
0,
.86602540378443864675,
-.86602540378443864675
};

const POSVEL_T sphdesign<3>::z[] = {
0,
0,
0
};

const POSVEL_T sphdesign<4>::x[] = {
.577350269189625763,
.577350269189625763,
-.577350269189625763,
-.577350269189625763
};

const POSVEL_T sphdesign<4>::y[] = {
.577350269189625763,
-.577350269189625763,
.577350269189625763,
-.577350269189625763
};

const POSVEL_T sphdesign<4>::z[] = {
.577350269189625763,
-.577350269189625763,
-.577350269189625763,
.577350269189625763
};

const POSVEL_T sphdesign<6>::x[] = {
1.0,
-1.0,
0,
0,
0,
0
};

const POSVEL_T sphdesign<6>::y[] = {
0,
0,
1.0,
-1.0,
0,
0
};

const POSVEL_T sphdesign<6>::z[] = {
0,
0,
0,
0,
1.0,
-1.0
};

const POSVEL_T sphdesign<12>::x[] = {
0,
0,
0.525731112119134,
-0.525731112119134,
0.85065080835204,
-0.85065080835204,
0,
0,
-0.525731112119134,
0.525731112119134,
-0.85065080835204,
0.85065080835204
};

const POSVEL_T sphdesign<12>::y[] = {
0.85065080835204,
0.85065080835204,
0,
0,
0.525731112119134,
0.525731112119134,
-0.85065080835204,
-0.85065080835204,
0,
0,
-0.525731112119134,
-0.525731112119134
};

const POSVEL_T sphdesign<12>::z[] = {
0.525731112119134,
-0.525731112119134,
0.85065080835204,
0.85065080835204,
0,
0,
-0.525731112119134,
0.525731112119134,
-0.85065080835204,
-0.85065080835204,
0,
0
};

const POSVEL_T sphdesign<14>::x[] = {
1.0e0,
5.947189772040725e-1,
5.947189772040725e-1,
5.947189772040725e-1,
-5.947189772040725e-1,
-5.947189772040725e-1,
-5.947189772040725e-1,
3.012536847870683e-1,
3.012536847870683e-1,
3.012536847870683e-1,
-3.012536847870683e-1,
-3.012536847870683e-1,
-3.012536847870683e-1,
-1.0e0
};

const POSVEL_T sphdesign<14>::y[] = {
0.0e0,
1.776539926025823e-1,
-7.678419429698292e-1,
5.90187950367247e-1,
1.776539926025823e-1,
5.90187950367247e-1,
-7.678419429698292e-1,
8.79474443923065e-1,
-7.588425179318781e-1,
-1.206319259911869e-1,
8.79474443923065e-1,
-1.206319259911869e-1,
-7.588425179318781e-1,
0.0e0
};

const POSVEL_T sphdesign<14>::z[] = {
0.0e0,
7.840589244857197e-1,
-2.381765915652909e-1,
-5.458823329204288e-1,
-7.840589244857197e-1,
5.458823329204288e-1,
2.381765915652909e-1,
3.684710570566285e-1,
5.774116818882528e-1,
-9.458827389448813e-1,
-3.684710570566285e-1,
9.458827389448813e-1,
-5.774116818882528e-1,
0.0e0
};
} 


template <int TDPTS>
RCBForceTree<TDPTS>::RCBForceTree(
#ifdef SYCL_LANGUAGE_VERSION
sycl::queue &sycl_queue,
#endif
POSVEL_T* minLoc,
POSVEL_T* maxLoc,
POSVEL_T* minForceLoc,
POSVEL_T* maxForceLoc,
ID_T count,
POSVEL_T* xLoc,
POSVEL_T* yLoc,
POSVEL_T* zLoc,
POSVEL_T* xVel,
POSVEL_T* yVel,
POSVEL_T* zVel,
POSVEL_T* ms,
POSVEL_T* phiLoc,
ID_T *idLoc,
MASK_T *maskLoc,
POSVEL_T avgMass,
POSVEL_T fsm,
POSVEL_T r,
POSVEL_T oa,
ID_T nd,
ID_T ds,
ID_T tmin,
ForceLaw *fl,
float fcoeff,
POSVEL_T ppc )
#ifdef SYCL_LANGUAGE_VERSION
: q(sycl_queue)
#endif
{
particleCount = count;

xx = xLoc;
yy = yLoc;
zz = zLoc;
vx = xVel;
vy = yVel;
vz = zVel;
mass = ms;

numThreads=1;

#define VMAX ALIGNY(16384)
nx_v = sycl::malloc_host<POSVEL_T>(VMAX * numThreads, q);
ny_v = sycl::malloc_host<POSVEL_T>(VMAX * numThreads, q);
nz_v = sycl::malloc_host<POSVEL_T>(VMAX * numThreads, q);
nm_v = sycl::malloc_host<POSVEL_T>(VMAX * numThreads, q);
for(int i = 0; i < VMAX*numThreads; i++) {
nx_v[i] = 0;
ny_v[i] = 0;
nz_v[i] = 0;
nm_v[i] = 0;
}

#ifdef SYCL_LANGUAGE_VERSION
int size=ALIGNY(nd);
d_xx = sycl::malloc_host<POSVEL_T>(size * numThreads, q);
d_yy = sycl::malloc_host<POSVEL_T>(size * numThreads, q);
d_zz = sycl::malloc_host<POSVEL_T>(size * numThreads, q);
d_vx = sycl::malloc_host<POSVEL_T>(size * numThreads, q);
d_vy = sycl::malloc_host<POSVEL_T>(size * numThreads, q);
d_vz = sycl::malloc_host<POSVEL_T>(size * numThreads, q);
d_mass = sycl::malloc_host<POSVEL_T>(size * numThreads, q);

for(int i = 0; i < size*numThreads; i++) {
d_xx[i] = 0;
d_yy[i] = 0;
d_zz[i] = 0;
d_vx[i] = 0;
d_vy[i] = 0;
d_vz[i] = 0;
d_mass[i] = 0;
}

d_nx_v = sycl::malloc_host<POSVEL_T>(VMAX * numThreads, q);
d_ny_v = sycl::malloc_host<POSVEL_T>(VMAX * numThreads, q);
d_nz_v = sycl::malloc_host<POSVEL_T>(VMAX * numThreads, q);
d_nm_v = sycl::malloc_host<POSVEL_T>(VMAX * numThreads, q);

for(int i = 0; i < VMAX*numThreads; i++) {
d_nx_v[i] = 0;
d_ny_v[i] = 0;
d_nz_v[i] = 0;
d_nm_v[i] = 0;
}

#endif

phi = phiLoc;
id = idLoc;
mask = maskLoc;

particleMass = avgMass;
fsrrmax = fsm;
rsm = r;
sinOpeningAngle = sinf(oa);
tanOpeningAngle = tanf(oa);
nDirect = nd;
depthSafety = ds;
taskPartMin = tmin;
ppContract = ppc;

for (int dim = 0; dim < DIMENSION; dim++) {
minRange[dim] = minLoc[dim];
maxRange[dim] = maxLoc[dim];
minForceRange[dim] = minForceLoc[dim];
maxForceRange[dim] = maxForceLoc[dim];
}

if (fl) {
m_own_fl = false;
m_fl = fl;
m_fcoeff = fcoeff;
} else {
m_own_fl = true;
m_fl = new ForceLawNewton();
m_fcoeff = 1.0;
}

ID_T nds = (((ID_T)(particleCount/(POSVEL_T)nDirect)) << depthSafety) + 1;
tree.reserve(nds);

int nthreads = 1;

timespec b_start, b_end;
clock_gettime(CLOCK_THREAD_CPUTIME_ID, &b_start);

createRCBForceTree();

clock_gettime(CLOCK_THREAD_CPUTIME_ID, &b_end);
double b_time = (b_end.tv_sec - b_start.tv_sec);
b_time += 1e-9*(b_end.tv_nsec - b_start.tv_nsec);

printStats(b_time);

inx.resize(nthreads);
iny.resize(nthreads);
inz.resize(nthreads);
inm.resize(nthreads);
iq.resize(nthreads);

calcInternodeForces(
#ifdef SYCL_LANGUAGE_VERSION
q
#endif
);
}

template <int TDPTS>
RCBForceTree<TDPTS>::~RCBForceTree()
{
if (m_own_fl) {
delete m_fl;
}
#ifdef SYCL_LANGUAGE_VERSION
sycl::free(d_xx, q);
sycl::free(d_yy, q);
sycl::free(d_zz, q);
sycl::free(d_vx, q);
sycl::free(d_vy, q);
sycl::free(d_vz, q);
sycl::free(d_mass, q);
sycl::free(d_nx_v, q);
sycl::free(d_ny_v, q);
sycl::free(d_nz_v, q);
sycl::free(d_nm_v, q);
#endif
sycl::free(nx_v, q);
sycl::free(ny_v, q);
sycl::free(nz_v, q);
sycl::free(nm_v, q);
}

template <int TDPTS>
void RCBForceTree<TDPTS>::printStats(double buildTime)
{
size_t zeroLeafNodes = 0;
size_t nonzeroLeafNodes = 0;
size_t maxPPN = 0;
size_t leafParts = 0;

for (ID_T tl = 1; tl < (ID_T) tree.size(); ++tl) {
if (tree[tl].cl == 0 && tree[tl].cr == 0) {
if (tree[tl].count > 0) {
++nonzeroLeafNodes;

leafParts += tree[tl].count;
maxPPN = std::max((size_t) tree[tl].count, maxPPN);
} else {
++zeroLeafNodes;
}
}
}

double localParticleCount = particleCount;
double localTreeSize = tree.size();
double localTreeCapacity = tree.capacity();
double localLeaves = zeroLeafNodes+nonzeroLeafNodes;
double localEmptyLeaves = zeroLeafNodes;
double localMeanPPN = leafParts/((double) nonzeroLeafNodes);
unsigned long localMaxPPN = maxPPN;
double localBuildTime = buildTime;

if ( Partition::getMyProc() == 0 ) {
printf("\ttree post-build statistics (local for rank 0):\n");
printf("\t\tparticles: %.2f\n", localParticleCount);
printf("\t\tnodes: %.2f (allocated:  %.2f)\n", localTreeSize, localTreeCapacity);
printf("\t\tleaves: %.2f (empty: %.2f)\n", localLeaves, localEmptyLeaves);
printf("\t\tmean ppn: %.2f (max ppn: %lu)\n", localMeanPPN, localMaxPPN);
printf("\t\tbuild time: %g s\n", localBuildTime);
}
}

void cm( ID_T count,
const POSVEL_T *xx, 
const POSVEL_T *yy,
const POSVEL_T *zz,
const POSVEL_T *mass,
POSVEL_T *xmin,
POSVEL_T *xmax,
POSVEL_T *xc)
{

double x = 0, y = 0, z = 0, m = 0;

for (ID_T i = 0; i < count; ++i) {
if (i == 0) {
xmin[0] = xmax[0] = xx[0];
xmin[1] = xmax[1] = yy[0];
xmin[2] = xmax[2] = zz[0];
} else {
xmin[0] = fminf(xmin[0], xx[i]);
xmax[0] = fmaxf(xmax[0], xx[i]);
xmin[1] = fminf(xmin[1], yy[i]);
xmax[1] = fmaxf(xmax[1], yy[i]);
xmin[2] = fminf(xmin[2], zz[i]);
xmax[2] = fmaxf(xmax[2], zz[i]);
}

POSVEL_T w = mass[i];
x += w*xx[i];
y += w*yy[i];
z += w*zz[i];
m += w;
}

xc[0] = (POSVEL_T) (x/m);
xc[1] = (POSVEL_T) (y/m);
xc[2] = (POSVEL_T) (z/m);
}

static inline POSVEL_T pptdr(const POSVEL_T*  xmin, const POSVEL_T*  xmax, const POSVEL_T*  xc)
{
return std::min(
xmax[0] - xc[0],
std::min(xmax[1] - xc[1],
std::min(xmax[2] - xc[2],
std::min(xc[0] - xmin[0],
std::min(xc[1] - xmin[1], xc[2] - xmin[2])))));
}

template <int TDPTS>
static inline void pppts(POSVEL_T tdr, const POSVEL_T*  xc,
POSVEL_T*  ppx, POSVEL_T*  ppy, POSVEL_T*  ppz)
{
for (int i = 0; i < TDPTS; ++i) {
ppx[i] = tdr*sphdesign<TDPTS>::x[i] + xc[0];
ppy[i] = tdr*sphdesign<TDPTS>::y[i] + xc[1];
ppz[i] = tdr*sphdesign<TDPTS>::z[i] + xc[2];
}
}

template <int TDPTS>
static inline void pp(ID_T count, const POSVEL_T*  xx, const POSVEL_T*  yy,
const POSVEL_T*  zz, const POSVEL_T*  mass, const POSVEL_T*  xc,
const POSVEL_T*  ppx, const POSVEL_T*  ppy, const POSVEL_T*  ppz,
POSVEL_T*  ppm, POSVEL_T tdr)
{
POSVEL_T K = TDPTS;
POSVEL_T odr0 = 1/K;

for (int i = 0; i < count; ++i) {
POSVEL_T xi = xx[i] - xc[0];
POSVEL_T yi = yy[i] - xc[1];
POSVEL_T zi = zz[i] - xc[2];
POSVEL_T ri = sqrtf(xi*xi + yi*yi + zi*zi);

for (int j = 0; j < TDPTS; ++j) {
POSVEL_T xj = ppx[j] - xc[0];
POSVEL_T yj = ppy[j] - xc[1];
POSVEL_T zj = ppz[j] - xc[2];
POSVEL_T rj2 = xj*xj + yj*yj + zj*zj;

POSVEL_T odr1 = 0, odr2 = 0;
if (rj2 != 0) {
POSVEL_T rj  = sqrtf(rj2);
POSVEL_T aij = (xi*xj + yi*yj + zi*zj)/(ri*rj);

odr1 = (3/K)*(ri/tdr)*aij;
odr2 = (5/K)*(ri/tdr)*(ri/tdr)*0.5*(3*aij*aij - 1);
}

ppm[j] += mass[i]*(odr0 + odr1 + odr2);
}
}
}

#ifdef SYCL_LANGUAGE_VERSION

typedef long long int int64;

template <typename T> inline T load(T *t)
{
return *t; 
}

template<int TILE_SIZE, typename T>void loadT(T *  out, const T *  in);

template <int TILE_SIZE, typename T>
inline void loadT(T *out, const T *in) {
#pragma unroll
for(int i=0;i<TILE_SIZE;i++) {
out[i] = *(in + i);
}
}

template <> inline void loadT<2, float>(float *out, const float *in) {
*reinterpret_cast<sycl::float2 *>(out) =
load(reinterpret_cast<const sycl::float2 *>(in));
}
template <> inline void loadT<4, float>(float *out, const float *in) {
*reinterpret_cast<sycl::float4 *>(out) =
load(reinterpret_cast<const sycl::float4 *>(in));
}

template <int TX, int TY>
inline void
computeForces(POSVEL_T xxi[], POSVEL_T yyi[], POSVEL_T zzi[], POSVEL_T xxj[],
POSVEL_T yyj[], POSVEL_T zzj[], POSVEL_T massj[], POSVEL_T xi[],
POSVEL_T yi[], POSVEL_T zi[], POSVEL_T ma0, POSVEL_T ma1,
POSVEL_T ma2, POSVEL_T ma3, POSVEL_T ma4, POSVEL_T ma5,
POSVEL_T mp_rsm2, POSVEL_T fsrrmax2) {

#pragma unroll
for(int i=0;i<TY;i++) {
#pragma unroll
for(int j=0;j<TX;j++) {
POSVEL_T dxc = xxj[j] - xxi[i];                                                                
POSVEL_T dyc = yyj[j] - yyi[i];                                                                
POSVEL_T dzc = zzj[j] - zzi[i];                                                                

POSVEL_T r2 = dxc * dxc + dyc * dyc + dzc * dzc;                                               
POSVEL_T v=r2+mp_rsm2;                                                                         
POSVEL_T v3=v*v*v;                                                                             


POSVEL_T f = sycl::rsqrt(v3); 

#ifndef BUG
f*=massj[j]*(r2<fsrrmax2 && r2>0.0f);                                                          
#else
f*=massj[j];                                                                                   
f*=(r2<fsrrmax2 && r2>0.0f);                                                                   
#endif

xi[i] = xi[i] + f * dxc;                                                                       
yi[i] = yi[i] + f * dyc;                                                                       
zi[i] = zi[i] + f * dzc;                                                                       
}
}
}

template <bool checkBounds, bool loadMass, int TILE_SIZE>
inline void
loadTile(int i, int bounds, const POSVEL_T *xx, const POSVEL_T *yy,
const POSVEL_T *zz, const POSVEL_T *mass, POSVEL_T xxi[],
POSVEL_T yyi[], POSVEL_T zzi[], POSVEL_T massi[]) {
if(checkBounds) {
#pragma unroll
for(int64 u=0;u<TILE_SIZE;u++) {
int64 idx=TILE_SIZE*i+u;                                                                        

#if 1
bool cond=idx<bounds;
xxi[u] = (cond) ? load(xx+idx) : 0.0f;                                                     
yyi[u] = (cond) ? load(yy+idx) : 0.0f;                                                     
zzi[u] = (cond) ? load(zz+idx) : 0.0f;                                                     
if(loadMass) massi[u] = (cond) ? load(mass+idx) : 0.0f;                                    
#else
massi[u] = 0.0f;                                                                           
if(idx<bounds) {                                                                           
xxi[u] = load(xx+idx);                                                                   
yyi[u] = load(yy+idx);                                                                   
zzi[u] = load(zz+idx);                                                                   
if(loadMass) massi[u] = load(mass+idx);                                                  
}
#endif
}
} else {

int idx=TILE_SIZE*i;
loadT<TILE_SIZE>(xxi,xx+idx);                                                                
loadT<TILE_SIZE>(yyi,yy+idx);                                                                
loadT<TILE_SIZE>(zzi,zz+idx);                                                                
if(loadMass) loadT<TILE_SIZE>(massi,mass+idx);                                               
}
}

template <bool checkBounds, int TILE_SIZE>
inline void applyForce(int i, int bounds, POSVEL_T fcoeff,
const POSVEL_T xi[],
const POSVEL_T yi[],
const POSVEL_T zi[],
POSVEL_T *vx,
POSVEL_T *vy,
POSVEL_T *vz,
sycl::nd_item<2> &item)
{
#pragma unroll
for(int u=0;u<TILE_SIZE;u++) {
int idx=TILE_SIZE*i+u;                                                                         

if(!checkBounds || idx<bounds)
{                                                                                           
atomicWarpReduceAndUpdate(vx + idx, fcoeff * xi[u], item); 
atomicWarpReduceAndUpdate(vy + idx, fcoeff * yi[u], item); 
atomicWarpReduceAndUpdate(vz + idx, fcoeff * zi[u], item); 
}
}
}


void Step10_kernel(int count, int count1,
const POSVEL_T*__restrict  xx, const POSVEL_T* __restrict yy,
const POSVEL_T*__restrict  zz, const POSVEL_T* __restrict mass,
const POSVEL_T*__restrict  xx1, const POSVEL_T*__restrict  yy1,
const POSVEL_T*__restrict  zz1, const POSVEL_T*__restrict  mass1,
POSVEL_T*__restrict  vx, POSVEL_T*__restrict  vy, POSVEL_T*__restrict  vz,
POSVEL_T fsrrmax2, POSVEL_T mp_rsm2, POSVEL_T fcoeff,
sycl::nd_item<2> &item)
{
const POSVEL_T ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, ma3 = -0.00109313, ma4 = 0.0000605491, ma5 = -0.00000147177;

POSVEL_T xxi[TILEY];
POSVEL_T yyi[TILEY];
POSVEL_T zzi[TILEY];
POSVEL_T xxj[TILEX];
POSVEL_T yyj[TILEX];
POSVEL_T zzj[TILEX];
POSVEL_T massj[TILEX];

int x_idx = item.get_global_id(1);
int y_idx = item.get_global_id(0);

for (int i = y_idx; i < count / TILEY;
i += item.get_local_range(0) * item.get_group_range(0)) 
{
POSVEL_T xi[TILEY]={0};                                                                                
POSVEL_T yi[TILEY]={0};                                                                                
POSVEL_T zi[TILEY]={0};                                                                                

loadTile<false,false,TILEY>(i,count,xx,yy,zz,NULL,xxi,yyi,zzi,NULL);

for (int j = x_idx; j < count1 / TILEX;
j += item.get_local_range(1) * item.get_group_range(1)) 
{
loadTile<false,true,TILEX>(j,count1,xx1,yy1,zz1,mass1,xxj,yyj,zzj,massj);

computeForces<TILEX,TILEY>(xxi,yyi,zzi,xxj,yyj,zzj,massj,xi,yi,zi,ma0,ma1,ma2,ma3,ma4,ma5,mp_rsm2,fsrrmax2);
}

for (int j = count1 / TILEX * TILEX + x_idx; j < count1;
j += item.get_local_range(1) * item.get_group_range(1)) 
{
loadTile<true,true,1>(j,count1,xx1,yy1,zz1,mass1,xxj,yyj,zzj,massj);

computeForces<1,TILEY>(xxi,yyi,zzi,xxj,yyj,zzj,massj,xi,yi,zi,ma0,ma1,ma2,ma3,ma4,ma5,mp_rsm2,fsrrmax2);
}

applyForce<false, TILEY>(i, count, fcoeff, xi, yi, zi, vx, vy, vz, item);
}


#if 1
for (int i = y_idx; i < count - count / TILEY * TILEY;
i += item.get_local_range(0) * item.get_group_range(0)) 
{
int k = i + count/TILEY*TILEY;
POSVEL_T xi[1]={0};                                                                                
POSVEL_T yi[1]={0};                                                                                
POSVEL_T zi[1]={0};                                                                                

loadTile<true,false,1>(k,count,xx,yy,zz,NULL,xxi,yyi,zzi,NULL);

for (int j = x_idx; j < count1 / TILEX;
j += item.get_local_range().get(1) * item.get_group_range(1)) 
{
loadTile<false,true,TILEX>(j,count1,xx1,yy1,zz1,mass1,xxj,yyj,zzj,massj);

computeForces<TILEX,1>(xxi,yyi,zzi,xxj,yyj,zzj,massj,xi,yi,zi,ma0,ma1,ma2,ma3,ma4,ma5,mp_rsm2,fsrrmax2);
}

for (int j = count1 / TILEX * TILEX + x_idx; j < count1;
j += item.get_local_range(1) * item.get_group_range(1)) 
{
loadTile<true,true,1>(j,count1,xx1,yy1,zz1,mass1,xxj,yyj,zzj,massj);

computeForces<1,1>(xxi,yyi,zzi,xxj,yyj,zzj,massj,xi,yi,zi,ma0,ma1,ma2,ma3,ma4,ma5,mp_rsm2,fsrrmax2);
}

applyForce<true, 1>(k, count, fcoeff, xi, yi, zi, vx, vy, vz, item);
}
#endif

}



#endif

#ifdef __bgq__
extern "C" Step16_int( int count1, float xxi, float yyi, float zzi, float fsrrmax2, float mp_rsm2, const float *xx1, const float *yy1, const float *zz1,const  float *mass1, float *ax, float *ay, float *az );
#endif

static inline void nbody1(ID_T count, ID_T count1, const POSVEL_T *xx,
const POSVEL_T *yy, const POSVEL_T *zz,
const POSVEL_T *mass, const POSVEL_T *xx1,
const POSVEL_T *yy1, const POSVEL_T *zz1,
const POSVEL_T *mass1, POSVEL_T *vx, POSVEL_T *vy,
POSVEL_T *vz, ForceLaw *fl, float fcoeff,
float fsrrmax, float rsm
#ifdef SYCL_LANGUAGE_VERSION
, sycl::queue &q
#endif
)
{
POSVEL_T fsrrmax2 = fsrrmax*fsrrmax;
POSVEL_T rsm2 = rsm*rsm;

#ifdef __bgq__
float ax = 0.0f, ay = 0.0f, az = 0.0f;

for (int i = 0; i < count; ++i)
{

Step16_int ( count1, xx[i],yy[i],zz[i], fsrrmax2,rsm2,xx1,yy1,zz1,mass1, &ax, &ay, &az );

vx[i] = vx[i] + ax * fcoeff;
vy[i] = vy[i] + ay * fcoeff;
vz[i] = vz[i] + az * fcoeff;
}

#else

#ifdef SYCL_LANGUAGE_VERSION

sycl::range<2> threads(BLOCKY, BLOCKX);
int blocksX = (count1 + threads[1] - 1) / threads[1];
int blocksY = (count + threads[0] - 1) / threads[0];
sycl::range<2> blocks(std::min(blocksY, MAXY), std::min(blocksX, MAXX));
sycl::range<2> gws (blocks * threads);

q.submit([&] (sycl::handler &cgh) {
cgh.parallel_for(sycl::nd_range<2>(gws, threads),
[=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
Step10_kernel(count, count1, xx, yy, zz, mass, xx1, yy1, zz1,
mass1, vx, vy, vz, fsrrmax2, rsm2, fcoeff, item);
});
}).wait();

#else

for (int i = 0; i < count; ++i)
for (int j = 0; j < count1; ++j) {
POSVEL_T dx = xx1[j] - xx[i];
POSVEL_T dy = yy1[j] - yy[i];
POSVEL_T dz = zz1[j] - zz[i];
POSVEL_T dist2 = dx*dx + dy*dy + dz*dz;
POSVEL_T f_over_r = mass[i]*mass1[j] * fl->f_over_r(dist2);

POSVEL_T updateq = 1.0;
updateq *= (dist2 < fsrrmax2);

vx[i] += updateq*fcoeff*f_over_r*dx;
vy[i] += updateq*fcoeff*f_over_r*dy;
vz[i] += updateq*fcoeff*f_over_r*dz;
}
#endif 


#endif 
}


static inline ID_T partition(ID_T n,
POSVEL_T*  xx, POSVEL_T*  yy, POSVEL_T*  zz,
POSVEL_T*  vx, POSVEL_T*  vy, POSVEL_T*  vz,
POSVEL_T*  mass, POSVEL_T*  phi,
ID_T*  id, MASK_T*  mask, POSVEL_T pv
)
{
float t0, t1, t2, t3, t4, t5, t6, t7;
int32_t is, i, j;
long i0;
uint16_t i1;

int idx[n];

is = 0;
for ( i = 0; i < n; i = i + 1 )
{
if (xx[i] < pv)
{
idx[is] = i;
is = is + 1;
}
}

#pragma unroll 4
for ( j = 0; j < is; j++ )
{
i = idx[j];

t6 = mass[i]; mass[i] = mass[j]; mass[j] = t6;
t7 = phi [i]; phi [i] = phi [j]; phi [j] = t7;
i1 = mask[i]; mask[i] = mask[j]; mask[j] = i1;
i0 = id  [i]; id  [i] = id  [j]; id  [j] = i0;
}

#pragma unroll 4
for ( j = 0; j < is; j++ )
{
i = idx[j];

t0 = xx[i]; xx[i] = xx[j]; xx[j] = t0;
t1 = yy[i]; yy[i] = yy[j]; yy[j] = t1;
t2 = zz[i]; zz[i] = zz[j]; zz[j] = t2;
t3 = vx[i]; vx[i] = vx[j]; vx[j] = t3;
t4 = vy[i]; vy[i] = vy[j]; vy[j] = t4;
t5 = vz[i]; vz[i] = vz[j]; vz[j] = t5;
}

return is;
}

template <int TDPTS>
void RCBForceTree<TDPTS>::createRCBForceSubtree(int d, ID_T tl, ID_T tlcl, ID_T tlcr)
{
POSVEL_T *x1, *x2, *x3;
switch (d) {
case 0:
x1 = xx;
x2 = yy;
x3 = zz;
break;
case 1:
x1 = yy;
x2 = zz;
x3 = xx;
break;
default:
x1 = zz;
x2 = xx;
x3 = yy;
break;
}

#ifdef __bgq__
int tid = 0;

#endif
const bool geoSplit = false;
POSVEL_T split = geoSplit ? (tree[tl].xmax[d]+tree[tl].xmin[d])/2 : tree[tl].xc[d];
ID_T is = ::partition(tree[tl].count, x1 + tree[tl].offset, x2 + tree[tl].offset, x3 + tree[tl].offset,
vx + tree[tl].offset, vy + tree[tl].offset, vz + tree[tl].offset,
mass + tree[tl].offset, phi + tree[tl].offset,
id + tree[tl].offset, mask + tree[tl].offset, split
);

if (is == 0 || is == tree[tl].count) {
return;
}

tree[tlcl].count = is;
tree[tlcr].count = tree[tl].count - tree[tlcl].count;

if (tree[tlcl].count > 0) {
tree[tl].cl = tlcl;
tree[tlcl].offset = tree[tl].offset;
tree[tlcl].xmax[d] = split;

createRCBForceTreeInParallel(tlcl);
}

if (tree[tlcr].count > 0) {
tree[tl].cr = tlcr;
tree[tlcr].offset = tree[tl].offset + tree[tlcl].count;
tree[tlcr].xmin[d] = split;

createRCBForceTreeInParallel(tlcr);
}
}

template <int TDPTS>
void RCBForceTree<TDPTS>::createRCBForceTreeInParallel(ID_T tl)
{
ID_T cnt = tree[tl].count;
ID_T off = tree[tl].offset;

cm(cnt, xx + off, yy + off, zz + off, mass + off,
tree[tl].xmin, tree[tl].xmax, tree[tl].xc);

if (cnt <= nDirect) {
tree[tl].tdr = ppContract*::pptdr(tree[tl].xmin, tree[tl].xmax, tree[tl].xc);
memset(tree[tl].ppm, 0, sizeof(POSVEL_T)*TDPTS);
if (cnt > TDPTS) { 
POSVEL_T ppx[TDPTS], ppy[TDPTS], ppz[TDPTS];
pppts<TDPTS>(tree[tl].tdr, tree[tl].xc, ppx, ppy, ppz);
pp<TDPTS>(cnt, xx + off, yy + off, zz + off, mass + off, tree[tl].xc,
ppx, ppy, ppz, tree[tl].ppm, tree[tl].tdr);
}

return;
}

ID_T tlcl, tlcr;
{
tlcl = tree.size();
tlcr = tlcl+1;
size_t newSize = tlcr+1;
tree.resize(newSize);
}
memset(&tree[tlcl], 0, sizeof(TreeNode)*2);

for (int i = 0; i < DIMENSION; ++i) {
tree[tlcl].xmin[i] = tree[tl].xmin[i];
tree[tlcr].xmin[i] = tree[tl].xmin[i];
tree[tlcl].xmax[i] = tree[tl].xmax[i];
tree[tlcr].xmax[i] = tree[tl].xmax[i];
}

POSVEL_T xlen[DIMENSION];
for (int i = 0; i < DIMENSION; ++i) {
xlen[i] = tree[tl].xmax[i] - tree[tl].xmin[i];
}

int d;
if (xlen[0] > xlen[1] && xlen[0] > xlen[2]) {
d = 0; 
}
else if (xlen[1] > xlen[2]) {
d = 1; 
}
else {
d = 2; 
}

createRCBForceSubtree(d, tl, tlcl, tlcr);

POSVEL_T ppx[TDPTS], ppy[TDPTS], ppz[TDPTS];
tree[tl].tdr = ppContract*::pptdr(tree[tl].xmin, tree[tl].xmax, tree[tl].xc);
pppts<TDPTS>(tree[tl].tdr, tree[tl].xc, ppx, ppy, ppz);
memset(tree[tl].ppm, 0, sizeof(POSVEL_T)*TDPTS);

if (tree[tlcl].count > 0) {
if (tree[tlcl].count <= TDPTS) {
ID_T offc = tree[tlcl].offset;
pp<TDPTS>(tree[tlcl].count, xx + offc, yy + offc, zz + offc, mass + offc,
tree[tl].xc, ppx, ppy, ppz, tree[tl].ppm, tree[tl].tdr);
} else {
POSVEL_T ppxc[TDPTS], ppyc[TDPTS], ppzc[TDPTS];
pppts<TDPTS>(tree[tlcl].tdr, tree[tlcl].xc, ppxc, ppyc, ppzc);
pp<TDPTS>(TDPTS, ppxc, ppyc, ppzc, tree[tlcl].ppm, tree[tl].xc,
ppx, ppy, ppz, tree[tl].ppm, tree[tl].tdr);
}
}
if (tree[tlcr].count > 0) {
if (tree[tlcr].count <= TDPTS) {
ID_T offc = tree[tlcr].offset;
pp<TDPTS>(tree[tlcr].count, xx + offc, yy + offc, zz + offc, mass + offc,
tree[tl].xc, ppx, ppy, ppz, tree[tl].ppm, tree[tl].tdr);
} else {
POSVEL_T ppxc[TDPTS], ppyc[TDPTS], ppzc[TDPTS];
pppts<TDPTS>(tree[tlcr].tdr, tree[tlcr].xc, ppxc, ppyc, ppzc);
pp<TDPTS>(TDPTS, ppxc, ppyc, ppzc, tree[tlcr].ppm, tree[tl].xc,
ppx, ppy, ppz, tree[tl].ppm, tree[tl].tdr);
}
}
}

template <int TDPTS>
void RCBForceTree<TDPTS>::createRCBForceTree()
{
tree.resize(1);
memset(&tree[0], 0, sizeof(TreeNode));

tree[0].count = particleCount;
tree[0].offset = 0;

for (int i = 0; i < DIMENSION; ++i) {
tree[0].xmin[i] = minRange[i];
tree[0].xmax[i] = maxRange[i];
}

createRCBForceTreeInParallel();
}


template <int TDPTS>
void RCBForceTree<TDPTS>::calcInternodeForce(
ID_T tl,
const std::vector<ID_T> &parents
#ifdef SYCL_LANGUAGE_VERSION
, sycl::queue &stream
#endif
) 
{
POSVEL_T fsrrmax2 = fsrrmax*fsrrmax;
const TreeNode* tree_ = &tree[0];

int tid = 0;

std::vector<ID_T> &q = iq[tid];
q.clear();
q.push_back(0);

POSVEL_T *nx=nx_v+tid*VMAX;
POSVEL_T *ny=ny_v+tid*VMAX;
POSVEL_T *nz=nz_v+tid*VMAX;
POSVEL_T *nm=nm_v+tid*VMAX;

#ifdef SYCL_LANGUAGE_VERSION
POSVEL_T *d_nx=d_nx_v+tid*VMAX;
POSVEL_T *d_ny=d_ny_v+tid*VMAX;
POSVEL_T *d_nz=d_nz_v+tid*VMAX;
POSVEL_T *d_nm=d_nm_v+tid*VMAX;

int size=ALIGNY(nDirect);

POSVEL_T *d_xxl=d_xx+tid*size;
POSVEL_T *d_yyl=d_yy+tid*size;
POSVEL_T *d_zzl=d_zz+tid*size;
POSVEL_T *d_massl=d_mass+tid*size;
POSVEL_T *d_vxl=d_vx+tid*size;
POSVEL_T *d_vyl=d_vy+tid*size;
POSVEL_T *d_vzl=d_vz+tid*size;

stream.wait(); 
#endif

int SIZE = 0; 

while (!q.empty()) {
ID_T tln = q.back();
q.pop_back();

if (tln < tl) {
bool isParent = std::binary_search(parents.begin(), parents.end(), tln);
if (isParent) {
ID_T tlncr = tree_[tln].cr;
ID_T tlncl = tree_[tln].cl;

if (tlncl != tl && tlncl > 0 && tree_[tlncl].count > 0) {
q.push_back(tlncl);
}
if (tlncr != tl && tlncr > 0 && tree_[tlncr].count > 0) {
q.push_back(tlncr);
}

continue;
}
}

POSVEL_T dx = tree_[tln].xc[0] - tree_[tl].xc[0];
POSVEL_T dy = tree_[tln].xc[1] - tree_[tl].xc[1];
POSVEL_T dz = tree_[tln].xc[2] - tree_[tl].xc[2];
POSVEL_T dist2 = dx*dx + dy*dy + dz*dz;

POSVEL_T sx = tree_[tln].xmax[0]-tree_[tln].xmin[0];
POSVEL_T sy = tree_[tln].xmax[1]-tree_[tln].xmin[1];
POSVEL_T sz = tree_[tln].xmax[2]-tree_[tln].xmin[2];
POSVEL_T l2 = std::min(sx*sx, std::min(sy*sy, sz*sz)); 

POSVEL_T dtt2 = dist2*tanOpeningAngle*tanOpeningAngle;
bool looksBig;
if (l2 > dtt2) {
looksBig = true;
} else {
looksBig = false;
for (int i = 0; i < 2; ++i)
for (int j = 0; j < 2; ++j) {
POSVEL_T x1 = (i == 0 ? tree_[tln].xmin : tree_[tln].xmax)[0] - tree_[tl].xc[0];
POSVEL_T y1 = (j == 0 ? tree_[tln].xmin : tree_[tln].xmax)[1] - tree_[tl].xc[1];
POSVEL_T z1 = tree_[tln].xmin[2] - tree_[tl].xc[2];

POSVEL_T x2 = (i == 0 ? tree_[tln].xmax : tree_[tln].xmin)[0] - tree_[tl].xc[0];
POSVEL_T y2 = (j == 0 ? tree_[tln].xmax : tree_[tln].xmin)[1] - tree_[tl].xc[1];
POSVEL_T z2 = tree_[tln].xmax[2] - tree_[tl].xc[2];

const bool useRealOA = false;
if (useRealOA) {
POSVEL_T cx = y1*z2 - z1*y2;
POSVEL_T cy = z1*x2 - x1*z2;
POSVEL_T cz = x1*y2 - y1*x2;
if ((cx*cx + cy*cy + cz*cz) > sinOpeningAngle*sinOpeningAngle*
(x1*x1 + y1*y1 + z1*z1)*(x2*x2 + y2*y2 + z2*z2)
) {
looksBig = true;
break;
}
} else {
POSVEL_T ddx = x1 - x2, ddy = y1 - y2, ddz = z1 - z2;
POSVEL_T dh2 = ddx*ddx + ddy*ddy + ddz*ddz;
if (dh2 > dtt2) {
looksBig = true;
break;
}
}
}
}

if (!looksBig) {
if (dist2 > fsrrmax2) {
continue;
}

if (tree_[tln].count <= TDPTS) {
ID_T offn = tree_[tln].offset;
ID_T cntn = tree_[tln].count;

int start = SIZE;
SIZE = SIZE + cntn;
assert( SIZE < VMAX );

for ( int i = 0; i < cntn; ++i) {
nx[start + i] = xx[offn + i];
ny[start + i] = yy[offn + i];
nz[start + i] = zz[offn + i];
nm[start + i] = mass[offn + i];
}

continue;
}

int start = SIZE;
SIZE = SIZE + TDPTS;
assert( SIZE < VMAX );

pppts<TDPTS>(tree_[tln].tdr, tree_[tln].xc, &nx[start], &ny[start], &nz[start]);
for ( int i = 0; i < TDPTS; ++i) {
nm[start + i] = tree_[tln].ppm[i];
}

continue;
} else if (tree_[tln].cr == 0 && tree_[tln].cl == 0) {
ID_T offn = tree_[tln].offset;
ID_T cntn = tree_[tln].count;

int start = SIZE;
SIZE = SIZE + cntn;
assert( SIZE < VMAX );

for ( int i = 0; i < cntn; ++i) {
nx[start + i] = xx[offn + i];
ny[start + i] = yy[offn + i];
nz[start + i] = zz[offn + i];
nm[start + i] = mass[offn + i];
}

continue;
}


ID_T tlncr = tree_[tln].cr;
ID_T tlncl = tree_[tln].cl;

if (tlncl > 0 && tree_[tlncl].count > 0) {
bool close = true;
for (int i = 0; i < DIMENSION; ++i) {
POSVEL_T dist = 0;
if (tree_[tl].xmax[i] < tree_[tlncl].xmin[i]) {
dist = tree_[tlncl].xmin[i] - tree_[tl].xmax[i];
} else if (tree_[tl].xmin[i] > tree_[tlncl].xmax[i]) {
dist = tree_[tl].xmin[i] - tree_[tlncl].xmax[i];
}

if (dist > fsrrmax) {
close = false;
break;
}
}

if (close) q.push_back(tlncl);
}
if (tlncr > 0 && tree_[tlncr].count > 0) {
bool close = true;
for (int i = 0; i < DIMENSION; ++i) {
POSVEL_T dist = 0;
if (tree_[tl].xmax[i] < tree_[tlncr].xmin[i]) {
dist = tree_[tlncr].xmin[i] - tree_[tl].xmax[i];
} else if (tree_[tl].xmin[i] > tree_[tlncr].xmax[i]) {
dist = tree_[tl].xmin[i] - tree_[tlncr].xmax[i];
}

if (dist > fsrrmax) {
close = false;
break;
}
}

if (close) q.push_back(tlncr);
}
}

ID_T off = tree_[tl].offset;
ID_T cnt = tree_[tl].count;

int start = SIZE;
SIZE = SIZE + cnt;
assert( SIZE < VMAX );

for ( int i = 0; i < cnt; ++i) {
nx[start + i] = xx[off + i];
ny[start + i] = yy[off + i];
nz[start + i] = zz[off + i];
nm[start + i] = mass[off + i];
}

#ifdef SYCL_LANGUAGE_VERSION
stream.memcpy(d_nx, nx, sizeof(POSVEL_T) * SIZE);
stream.memcpy(d_ny, ny, sizeof(POSVEL_T) * SIZE);
stream.memcpy(d_nz, nz, sizeof(POSVEL_T) * SIZE);
stream.memcpy(d_nm, nm, sizeof(POSVEL_T) * SIZE);

stream.memcpy(d_xxl, xx + off, sizeof(POSVEL_T) * cnt);
stream.memcpy(d_yyl, yy + off, sizeof(POSVEL_T) * cnt);
stream.memcpy(d_zzl, zz + off, sizeof(POSVEL_T) * cnt);
stream.memcpy(d_massl, mass + off, sizeof(POSVEL_T) * cnt);
stream.memcpy(d_vxl, vx + off, sizeof(POSVEL_T) * cnt);
stream.memcpy(d_vyl, vy + off, sizeof(POSVEL_T) * cnt);
stream.memcpy(d_vzl, vz + off, sizeof(POSVEL_T) * cnt);
stream.wait();

::nbody1(cnt, SIZE, d_xxl, d_yyl, d_zzl, d_massl, d_nx, d_ny, d_nz, d_nm,
d_vxl, d_vyl, d_vzl, m_fl, m_fcoeff, fsrrmax, rsm, stream);
stream.wait();

stream.memcpy(vx + off, d_vxl, sizeof(POSVEL_T) * cnt);
stream.memcpy(vy + off, d_vyl, sizeof(POSVEL_T) * cnt);
stream.memcpy(vz + off, d_vzl, sizeof(POSVEL_T) * cnt);
stream.wait();
#else
::nbody1(cnt, SIZE, xx + off, yy + off, zz + off, mass + off, nx, ny, nz, nm, 
vx + off, vy + off, vz + off, m_fl, m_fcoeff, fsrrmax, rsm);
#endif
}

template <int TDPTS>
void RCBForceTree<TDPTS>::calcInternodeForces(
#ifdef SYCL_LANGUAGE_VERSION
sycl::queue &stream
#endif
)
{

std::vector<ID_T> q(1, 0);
std::vector<ID_T> parents;
while (!q.empty()) {
ID_T tl = q.back();
if (tree[tl].cr == 0 && tree[tl].cl == 0) {
q.pop_back();

bool inside = true;
for (int i = 0; i < DIMENSION; ++i) {
inside &= (tree[tl].xmax[i] < maxForceRange[i] && tree[tl].xmax[i] > minForceRange[i]) ||
(tree[tl].xmin[i] < maxForceRange[i] && tree[tl].xmin[i] > minForceRange[i]);
}

if (inside) {
calcInternodeForce(tl, parents
#ifdef SYCL_LANGUAGE_VERSION
, stream
#endif
);
}
} else if (parents.size() > 0 && parents.back() == tl) {
parents.pop_back();
q.pop_back();
} else {
if (tree[tl].cl > 0) q.push_back(tree[tl].cl);
if (tree[tl].cr > 0) q.push_back(tree[tl].cr);
parents.push_back(tl);
}
}
}

template class RCBForceTree<QUADRUPOLE_TDPTS>;
template class RCBForceTree<MONOPOLE_TDPTS>;
