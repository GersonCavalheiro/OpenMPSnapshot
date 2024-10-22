

#include "bf.h"


#if defined(SharedTBB) && defined(peanoCall)
std::vector<delta::contact::contactpoint> delta::contact::detection::bf(
const iREAL*    xCoordinatesOfPointsOfGeometryA,
const iREAL*    yCoordinatesOfPointsOfGeometryA,
const iREAL*    zCoordinatesOfPointsOfGeometryA,
int             numberOfPointsOfGeometryA,
iREAL           epsilonA,
bool            frictionA,
int 	          particleA,

const iREAL*    xCoordinatesOfPointsOfGeometryB,
const iREAL*    yCoordinatesOfPointsOfGeometryB,
const iREAL*    zCoordinatesOfPointsOfGeometryB,
int             numberOfPointsOfGeometryB,
iREAL           epsilonB,
bool            frictionB,
int 	          particleB,
tarch::multicore::BooleanSemaphore &semaphore)
#else
std::vector<delta::contact::contactpoint> delta::contact::detection::bf(
const iREAL*    xCoordinatesOfPointsOfGeometryA,
const iREAL*    yCoordinatesOfPointsOfGeometryA,
const iREAL*    zCoordinatesOfPointsOfGeometryA,
int             numberOfPointsOfGeometryA,
iREAL           epsilonA,
bool            frictionA,
int 	          particleA,

const iREAL*    xCoordinatesOfPointsOfGeometryB,
const iREAL*    yCoordinatesOfPointsOfGeometryB,
const iREAL*    zCoordinatesOfPointsOfGeometryB,
int             numberOfPointsOfGeometryB,
iREAL           epsilonB,
bool            frictionB,
int 	          particleB)
#endif
{
#if defined(__INTEL_COMPILER)
__assume_aligned(xCoordinatesOfPointsOfGeometryA, byteAlignment);
__assume_aligned(yCoordinatesOfPointsOfGeometryA, byteAlignment);
__assume_aligned(zCoordinatesOfPointsOfGeometryA, byteAlignment);

__assume_aligned(xCoordinatesOfPointsOfGeometryB, byteAlignment);
__assume_aligned(yCoordinatesOfPointsOfGeometryB, byteAlignment);
__assume_aligned(zCoordinatesOfPointsOfGeometryB, byteAlignment);
#endif

std::vector<contactpoint> result;

int numberOfTrianglesA = numberOfPointsOfGeometryA * 3;
int numberOfTrianglesB = numberOfPointsOfGeometryB * 3;

#if defined(SharedTBB) && defined(peanoCall)
tarch::multicore::Lock lock(semaphore,false);
#endif

#if defined(SharedTBB) && defined(peanoCall)
const int grainSize = numberOfTrianglesA;
tbb::parallel_for(
tbb::blocked_range<int>(0, numberOfTrianglesA, grainSize), [&](const tbb::blocked_range<int>& r)
{
for(std::vector<int>::size_type iA=0; iA<r.size(); iA+=3)
#else
#if defined(OMPTriangle) && defined(OMPProcess)
#pragma omp parallel for shared(result) firstprivate(numberOfTrianglesA, numberOfTrianglesB, epsilonA, epsilonB, frictionA, frictionB, particleA, particleB, xCoordinatesOfPointsOfGeometryA, yCoordinatesOfPointsOfGeometryA, zCoordinatesOfPointsOfGeometryA, xCoordinatesOfPointsOfGeometryB, yCoordinatesOfPointsOfGeometryB, zCoordinatesOfPointsOfGeometryB)
#endif
for(int iA=0; iA<numberOfTrianglesA; iA+=3)
#endif
{
__attribute__ ((aligned(byteAlignment))) iREAL xPA[10000], yPA[10000], zPA[10000], xPB[10000], yPB[10000], zPB[10000], d[10000];

#ifdef OMPProcess
#pragma omp simd
#endif
for(int iB=0; iB<numberOfTrianglesB; iB+=3)
{
bfSolver(	xCoordinatesOfPointsOfGeometryA+(iA),
yCoordinatesOfPointsOfGeometryA+(iA),
zCoordinatesOfPointsOfGeometryA+(iA),
xCoordinatesOfPointsOfGeometryB+(iB),
yCoordinatesOfPointsOfGeometryB+(iB),
zCoordinatesOfPointsOfGeometryB+(iB),
xPA[iB],
yPA[iB],
zPA[iB],
xPB[iB],
yPB[iB],
zPB[iB]);
d[iB] = std::sqrt(((xPB[iB]-xPA[iB])*(xPB[iB]-xPA[iB]))+((yPB[iB]-yPA[iB])*(yPB[iB]-yPA[iB]))+((zPB[iB]-zPA[iB])*(zPB[iB]-zPA[iB])));
}

__attribute__ ((aligned(byteAlignment))) std::vector<contactpoint> nearestContactPoint;
__attribute__ ((aligned(byteAlignment))) iREAL epsilonMargin = (epsilonA+epsilonB);

for(int iB=0; iB<numberOfTrianglesB; iB+=3)
{
if(d[iB] <= epsilonMargin)
{
nearestContactPoint.push_back(contactpoint(xPA[iB], yPA[iB], zPA[iB], epsilonA, particleA, xPB[iB], yPB[iB], zPB[iB], epsilonB, particleB, frictionA && frictionB));
}
}

#if defined(SharedTBB) && defined(peanoCall)
lock.lock();
for(int xx=0; xx < nearestContactPoint.size(); xx++)
result.push_back(nearestContactPoint[xx]);
lock.free();
#else
#ifdef OMPTriangle
#pragma omp critical
#endif
for(int xx=0; xx < nearestContactPoint.size(); xx++)
result.push_back(nearestContactPoint[xx]);
#endif
#if defined(SharedTBB) && defined(peanoCall)
}});
#else
}
#endif
return result;
}

#define DOT(a, b)\
((a)[0]*(b)[0] + (a)[1]*(b)[1] + (a)[2]*(b)[2])


#define CROSS(dest,v1,v2){                     \
dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
dest[2]=v1[0]*v2[1]-v1[1]*v2[0];}

int NoDivTriTriIsect(
iREAL V0[3],iREAL V1[3],iREAL V2[3],
iREAL U0[3],iREAL U1[3],iREAL U2[3])
{
iREAL E1[3],E2[3];
iREAL N1[3],N2[3],d1,d2;
iREAL du0,du1,du2,dv0,dv1,dv2;
iREAL D[3];
iREAL isect1[2], isect2[2];
iREAL du0du1,du0du2,dv0dv1,dv0dv2;
short index;
iREAL vp0,vp1,vp2;
iREAL up0,up1,up2;
iREAL bb,cc,max;


SUB(V1,V0, E1);
SUB(V2,V0, E2);
CROSS(E1, E2, N1);
d1=-DOT(N1,V0);



du0=DOT(N1,U0)+d1;
du1=DOT(N1,U1)+d1;
du2=DOT(N1,U2)+d1;


du0du1=du0*du1;
du0du2=du0*du2;

if(du0du1>0.0f && du0du2>0.0f)
return 1;                    


SUB(U1,U0,E1);
SUB(U2,U0,E2);
CROSS(E1,E2,N2);

d2=-DOT(N2,U0);



dv0=DOT(N2,V0)+d2;
dv1=DOT(N2,V1)+d2;
dv2=DOT(N2,V2)+d2;

dv0dv1=dv0*dv1;
dv0dv2=dv0*dv2;

if(dv0dv1>0.0f && dv0dv2>0.0f)
return 1;


return 0;
}

void delta::contact::detection::bfSolver(
const iREAL   *xxCoordinatesOfPointsOfGeometryA,
const iREAL   *yyCoordinatesOfPointsOfGeometryA,
const iREAL   *zzCoordinatesOfPointsOfGeometryA,
const iREAL   *xxCoordinatesOfPointsOfGeometryB,
const iREAL   *yyCoordinatesOfPointsOfGeometryB,
const iREAL   *zzCoordinatesOfPointsOfGeometryB,
iREAL&  		xPA,
iREAL&  		yPA,
iREAL&  		zPA,
iREAL&  		xPB,
iREAL&  		yPB,
iREAL&  		zPB)
{



{
}


iREAL u[3], v[3], nn[3][2], w[3], w0[3][6], dir[3][6], pointArray[3][6];

u[0] = xxCoordinatesOfPointsOfGeometryB[1] - xxCoordinatesOfPointsOfGeometryB[0];
u[1] = yyCoordinatesOfPointsOfGeometryB[1] - yyCoordinatesOfPointsOfGeometryB[0];
u[2] = zzCoordinatesOfPointsOfGeometryB[1] - zzCoordinatesOfPointsOfGeometryB[0];

v[0] = xxCoordinatesOfPointsOfGeometryB[2] - xxCoordinatesOfPointsOfGeometryB[1];
v[1] = yyCoordinatesOfPointsOfGeometryB[2] - yyCoordinatesOfPointsOfGeometryB[1];
v[2] = zzCoordinatesOfPointsOfGeometryB[2] - zzCoordinatesOfPointsOfGeometryB[1];



nn[0][0] = u[1]*v[2] - u[2]*v[1];
nn[1][0] = u[2]*v[0] - u[0]*v[2];
nn[2][0] = u[0]*v[1] - u[1]*v[0];


dir[0][0] = xxCoordinatesOfPointsOfGeometryA[1] - xxCoordinatesOfPointsOfGeometryA[0];
dir[1][0] = yyCoordinatesOfPointsOfGeometryA[1] - yyCoordinatesOfPointsOfGeometryA[0];
dir[2][0] = zzCoordinatesOfPointsOfGeometryA[1] - zzCoordinatesOfPointsOfGeometryA[0];

w0[0][0] = xxCoordinatesOfPointsOfGeometryA[0] - xxCoordinatesOfPointsOfGeometryB[0];
w0[1][0] = yyCoordinatesOfPointsOfGeometryA[0] - yyCoordinatesOfPointsOfGeometryB[0];
w0[2][0] = zzCoordinatesOfPointsOfGeometryA[0] - zzCoordinatesOfPointsOfGeometryB[0];

dir[0][1] = xxCoordinatesOfPointsOfGeometryA[2] - xxCoordinatesOfPointsOfGeometryA[1];
dir[1][1] = yyCoordinatesOfPointsOfGeometryA[2] - yyCoordinatesOfPointsOfGeometryA[1];
dir[2][1] = zzCoordinatesOfPointsOfGeometryA[2] - zzCoordinatesOfPointsOfGeometryA[1];

w0[0][1] = xxCoordinatesOfPointsOfGeometryA[1] - xxCoordinatesOfPointsOfGeometryB[0];
w0[1][1] = yyCoordinatesOfPointsOfGeometryA[1] - yyCoordinatesOfPointsOfGeometryB[0];
w0[2][1] = zzCoordinatesOfPointsOfGeometryA[1] - zzCoordinatesOfPointsOfGeometryB[0];

dir[0][2] = xxCoordinatesOfPointsOfGeometryA[0] - xxCoordinatesOfPointsOfGeometryA[2];
dir[1][2] = yyCoordinatesOfPointsOfGeometryA[0] - yyCoordinatesOfPointsOfGeometryA[2];
dir[2][2] = zzCoordinatesOfPointsOfGeometryA[0] - zzCoordinatesOfPointsOfGeometryA[2];

w0[0][2] = xxCoordinatesOfPointsOfGeometryA[2] - xxCoordinatesOfPointsOfGeometryB[0];
w0[1][2] = yyCoordinatesOfPointsOfGeometryA[2] - yyCoordinatesOfPointsOfGeometryB[0];
w0[2][2] = zzCoordinatesOfPointsOfGeometryA[2] - zzCoordinatesOfPointsOfGeometryB[0];

u[0] = xxCoordinatesOfPointsOfGeometryA[1] - xxCoordinatesOfPointsOfGeometryA[0];
u[1] = yyCoordinatesOfPointsOfGeometryA[1] - yyCoordinatesOfPointsOfGeometryA[0];
u[2] = zzCoordinatesOfPointsOfGeometryA[1] - zzCoordinatesOfPointsOfGeometryA[0];

v[0] = xxCoordinatesOfPointsOfGeometryA[2] - xxCoordinatesOfPointsOfGeometryA[0];
v[1] = yyCoordinatesOfPointsOfGeometryA[2] - yyCoordinatesOfPointsOfGeometryA[0];
v[2] = zzCoordinatesOfPointsOfGeometryA[2] - zzCoordinatesOfPointsOfGeometryA[0];

nn[0][1] = u[1]*v[2] - u[2]*v[1];
nn[1][1] = u[2]*v[0] - u[0]*v[2];
nn[2][1] = u[0]*v[1] - u[1]*v[0];

dir[0][3] = xxCoordinatesOfPointsOfGeometryB[1] - xxCoordinatesOfPointsOfGeometryB[0];
dir[1][3] = yyCoordinatesOfPointsOfGeometryB[1] - yyCoordinatesOfPointsOfGeometryB[0];
dir[2][3] = zzCoordinatesOfPointsOfGeometryB[1] - zzCoordinatesOfPointsOfGeometryB[0];

w0[0][3] = xxCoordinatesOfPointsOfGeometryB[0] - xxCoordinatesOfPointsOfGeometryA[0];
w0[1][3] = yyCoordinatesOfPointsOfGeometryB[0] - yyCoordinatesOfPointsOfGeometryA[0];
w0[2][3] = zzCoordinatesOfPointsOfGeometryB[0] - zzCoordinatesOfPointsOfGeometryA[0];

dir[0][4] = xxCoordinatesOfPointsOfGeometryB[2] - xxCoordinatesOfPointsOfGeometryB[1];
dir[1][4] = yyCoordinatesOfPointsOfGeometryB[2] - yyCoordinatesOfPointsOfGeometryB[1];
dir[2][4] = zzCoordinatesOfPointsOfGeometryB[2] - zzCoordinatesOfPointsOfGeometryB[1];

w0[0][4] = xxCoordinatesOfPointsOfGeometryB[1] - xxCoordinatesOfPointsOfGeometryA[0];
w0[1][4] = yyCoordinatesOfPointsOfGeometryB[1] - yyCoordinatesOfPointsOfGeometryA[0];
w0[2][4] = zzCoordinatesOfPointsOfGeometryB[1] - zzCoordinatesOfPointsOfGeometryA[0];

dir[0][5] = xxCoordinatesOfPointsOfGeometryB[0] - xxCoordinatesOfPointsOfGeometryB[2];
dir[1][5] = yyCoordinatesOfPointsOfGeometryB[0] - yyCoordinatesOfPointsOfGeometryB[2];
dir[2][5] = zzCoordinatesOfPointsOfGeometryB[0] - zzCoordinatesOfPointsOfGeometryB[2];

w0[0][5] = xxCoordinatesOfPointsOfGeometryB[2] - xxCoordinatesOfPointsOfGeometryA[0];
w0[1][5] = yyCoordinatesOfPointsOfGeometryB[2] - yyCoordinatesOfPointsOfGeometryA[0];
w0[2][5] = zzCoordinatesOfPointsOfGeometryB[2] - zzCoordinatesOfPointsOfGeometryA[0];

for(int j=0;j<6;j++)
{
iREAL a,b;
if(j<3)
{
a = -DOT(nn[0], w0[0]);
b= DOT(nn[0],dir[0]);
}else {
a = -DOT(nn[1], w0[0]);
b= DOT(nn[1],dir[0]);
}
if (abs(b) < 1E-30 && a==0) {
break; 
}

iREAL r = a / b;

xPA = pointArray[0][j] + r * dir[0][j];
yPA = pointArray[1][j] + r * dir[1][j];
zPA = pointArray[2][j] + r * dir[2][j];

xPB = xPA;
yPB = yPA;
zPB = zPA;

if (r < 0.0 || r > 1.0 || r != r) { break;}

if(j<3){
w[0] = xPA - xxCoordinatesOfPointsOfGeometryB[0];
w[1] = yPA - yyCoordinatesOfPointsOfGeometryB[0];
w[2] = zPA - zzCoordinatesOfPointsOfGeometryB[0];
}else{
w[0] = xPA - xxCoordinatesOfPointsOfGeometryA[0];
w[1] = yPA - yyCoordinatesOfPointsOfGeometryA[0];
w[2] = zPA - zzCoordinatesOfPointsOfGeometryA[0];
}

iREAL D = DOT(u,v)*DOT(u,v) - DOT(u,u)*DOT(v,v);
iREAL s = (DOT(u,v)*DOT(w,v) - DOT(v,v)*DOT(w,u))/D;
if (s<0.0 || s>1.0) {
break;
}

iREAL t = (DOT(u,v)*DOT(w,u) - DOT(u,u)*DOT(w,v))/D;
if (t<0.0 || (s+t)>1.0)
{
break;
}
}

iREAL a[9], b[9], c[9], d[9], e[9], f[9], p1[3], p2[3], p3[3], p4[3], p5[3], p6[3];

p1[0] = xxCoordinatesOfPointsOfGeometryA[0];
p1[1] = yyCoordinatesOfPointsOfGeometryA[0];
p1[2] = zzCoordinatesOfPointsOfGeometryA[0];

p2[0] = xxCoordinatesOfPointsOfGeometryA[1];
p2[1] = yyCoordinatesOfPointsOfGeometryA[1];
p2[2] = zzCoordinatesOfPointsOfGeometryA[1];

p3[0] = xxCoordinatesOfPointsOfGeometryA[2];
p3[1] = yyCoordinatesOfPointsOfGeometryA[2];
p3[2] = zzCoordinatesOfPointsOfGeometryA[2];

p4[0] = xxCoordinatesOfPointsOfGeometryB[0];
p4[1] = yyCoordinatesOfPointsOfGeometryB[0];
p4[2] = zzCoordinatesOfPointsOfGeometryB[0];

p5[0] = xxCoordinatesOfPointsOfGeometryB[1];
p5[1] = yyCoordinatesOfPointsOfGeometryB[1];
p5[2] = zzCoordinatesOfPointsOfGeometryB[1];

p6[0] = xxCoordinatesOfPointsOfGeometryB[2];
p6[1] = yyCoordinatesOfPointsOfGeometryB[2];
p6[2] = zzCoordinatesOfPointsOfGeometryB[2];


u[0] = (p1[0] - p2[0]);
u[1] = (p1[1] - p2[1]);
u[2] = (p1[2] - p2[2]);


v[0] = (p4[0] - p5[0]);
v[1] = (p4[1] - p5[1]);
v[2] = (p4[2] - p5[2]);


w[0] = (p2[0] - p5[0]);
w[1] = (p2[1] - p5[1]);
w[2] = (p2[2] - p5[2]);

a[0] = DOT(u,u);
b[0] = DOT(u,v);
c[0] = DOT(v,v);
d[0] = DOT(u,w);
e[0] = DOT(v,w);

v[0] = (p5[0] - p6[0]);
v[1] = (p5[1] - p6[1]);
v[2] = (p5[2] - p6[2]);

w[0] = (p2[0] - p6[0]);
w[1] = (p2[1] - p6[1]);
w[2] = (p2[2] - p6[2]);

a[1] = a[0];
b[1] = DOT(u,v);
c[1] = DOT(v,v);
d[1] = DOT(u,w);
e[1] = DOT(v,w);

v[0] = (p6[0] - p4[0]);
v[1] = (p6[1] - p4[1]);
v[2] = (p6[2] - p4[2]);


w[0] = (p2[0] - p4[0]);
w[1] = (p2[1] - p4[1]);
w[2] = (p2[2] - p4[2]);

a[2] = a[0];
b[2] = DOT(u,v);
c[2] = DOT(v,v);
d[2] = DOT(u,w);
e[2] = DOT(v,w);

u[0] = (p2[0] - p3[0]);
u[1] = (p2[1] - p3[1]);
u[2] = (p2[2] - p3[2]);


v[0] = (p4[0] - p5[0]);
v[1] = (p4[1] - p5[1]);
v[2] = (p4[2] - p5[2]);

w[0] = (p3[0] - p5[0]);
w[1] = (p3[1] - p5[1]);
w[2] = (p3[2] - p5[2]);

a[3] = DOT(u,u);
b[3] = DOT(u,v);
c[3] = DOT(v,v);
d[3] = DOT(u,w);
e[3] = DOT(v,w);

v[0] = (p5[0] - p6[0]);
v[1] = (p5[1] - p6[1]);
v[2] = (p5[2] - p6[2]);


w[0] = (p3[0] - p6[0]);
w[1] = (p3[1] - p6[1]);
w[2] = (p3[2] - p6[2]);

a[4] = a[3];
b[4] = DOT(u,v);
c[4] = DOT(v,v);
d[4] = DOT(u,w);
e[4] = DOT(v,w);

v[0] = (p6[0] - p4[0]);
v[1] = (p6[1] - p4[1]);
v[2] = (p6[2] - p4[2]);

w[0] = (p3[0] - p4[0]);
w[1] = (p3[1] - p4[1]);
w[2] = (p3[2] - p4[2]);

a[5] = a[3];
b[5] = DOT(u,v);
c[5] = DOT(v,v);
d[5] = DOT(u,w);
e[5] = DOT(v,w);


u[0] = (p3[0] - p1[0]);
u[1] = (p3[1] - p1[1]);
u[2] = (p3[2] - p1[2]);

v[0] = (p4[0] - p5[0]);
v[1] = (p4[1] - p5[1]);
v[2] = (p4[2] - p5[2]);

w[0] = (p1[0] - p5[0]);
w[1] = (p1[1] - p5[1]);
w[2] = (p1[2] - p5[2]);

a[6] = DOT(u,u);
b[6] = DOT(u,v);
c[6] = DOT(v,v);
d[6] = DOT(u,w);
e[6] = DOT(v,w);

v[0] = (p5[0] - p6[0]);
v[1] = (p5[1] - p6[1]);
v[2] = (p5[2] - p6[2]);

w[0] = (p1[0] - p6[0]);
w[1] = (p1[1] - p6[1]);
w[2] = (p1[2] - p6[2]);

a[7] = a[6];
b[7] = DOT(u,v);
c[7] = DOT(v,v);
d[7] = DOT(u,w);
e[7] = DOT(v,w);

v[0] = (p6[0] - p4[0]);
v[1] = (p6[1] - p4[1]);
v[2] = (p6[2] - p4[2]);

w[0] = (p1[0] - p4[0]);
w[1] = (p1[1] - p4[1]);
w[2] = (p1[2] - p4[2]);

a[8] = a[6];
b[8] = DOT(u,v);
c[8] = DOT(v,v);
d[8] = DOT(u,w);
e[8] = DOT(v,w);


iREAL ssmin = 1E+30, ttc=0, ssc=0;
int ssid = 0;

for( int j=0; j<9;j++)
{
iREAL D = (a[j]*c[j] - b[j]*b[j]);

iREAL sD = D;
iREAL tD = D;
iREAL sN = 0, tN = 0;

iREAL SMALL_NUM = 1E-30;

if (D < SMALL_NUM)
{
sN = 0.0;       
sD = 1.0;       
tN = e[j];
tD = c[j];
} else {               
sN = (b[j]*e[j] - c[j]*d[j]);
tN = (a[j]*e[j] - b[j]*d[j]);

if (sN < 0.0){   
sN = 0.0;
tN = e[j];
tD = c[j];
} else if (sN > sD){
sN = sD;
tN = e[j] + b[j];
tD = c[j];
}
}

if (tN < 0.0){     
tN = 0.0; 
if (-d[j] < 0.0){
sN = 0.0;
}else if (-d[j] > a[j]){
sN = sD;
} else {
sN = -d[j];
sD = a[j];
}
} else if (tN > tD){       
tN = tD;  
if ((-d[j] + b[j]) < 0.0){
sN = 0;
}else if ((-d[j] + b[j]) > a[j]){
sN = sD;
} else {
sN = (-d[j] + b[j]);
sD = a[j];
}
}

iREAL sc, tc; 
if(abs(sN) < SMALL_NUM){
sc = 0.0;
}else{
sc = sN / sD;
}

if(abs(tN) < SMALL_NUM){
tc = 0.0;
}else{
tc = tN / tD;
}

iREAL dP[3], dist;

if(j==0)
{
dP[0] = (p2[0] - p5[0]) + (sc*(p1[0] - p2[0])) - (tc*(p4[0] - p5[0]));
dP[1] = (p2[1] - p5[1]) + (sc*(p1[1] - p2[1])) - (tc*(p4[1] - p5[1]));
dP[2] = (p2[2] - p5[2]) + (sc*(p1[2] - p2[2])) - (tc*(p4[2] - p5[2]));
}else if(j==1)
{
dP[0] = (p2[0] - p6[0]) + (sc*(p1[0] - p2[0])) - (tc*(p5[0] - p6[0]));
dP[1] = (p2[1] - p6[1]) + (sc*(p1[1] - p2[1])) - (tc*(p5[1] - p6[1]));
dP[2] = (p2[2] - p6[2]) + (sc*(p1[2] - p2[2])) - (tc*(p5[2] - p6[2]));
}else if(j==2)
{
dP[0] = (p2[0] - p4[0]) + (sc*(p1[0] - p2[0])) - (tc*(p6[0] - p4[0]));
dP[1] = (p2[1] - p4[1]) + (sc*(p1[1] - p2[1])) - (tc*(p6[1] - p4[1]));
dP[2] = (p2[2] - p4[2]) + (sc*(p1[2] - p2[2])) - (tc*(p6[2] - p4[2]));
}else if(j==3)
{
dP[0] = (p3[0] - p5[0]) + (sc*(p2[0] - p3[0])) - (tc*(p4[0] - p5[0]));
dP[1] = (p3[1] - p5[1]) + (sc*(p2[1] - p3[1])) - (tc*(p4[1] - p5[1]));
dP[2] = (p3[2] - p5[2]) + (sc*(p2[2] - p3[2])) - (tc*(p4[2] - p5[2]));
}else if(j==4)
{
dP[0] = (p3[0] - p6[0]) + (sc*(p2[0] - p3[0])) - (tc*(p5[0] - p6[0]));
dP[1] = (p3[1] - p6[1]) + (sc*(p2[1] - p3[1])) - (tc*(p5[1] - p6[1]));
dP[2] = (p3[2] - p6[2]) + (sc*(p2[2] - p3[2])) - (tc*(p5[2] - p6[2]));
}else if(j==5)
{
dP[0] = (p3[0] - p4[0]) + (sc*(p2[0] - p3[0])) - (tc*(p6[0] - p4[0]));
dP[1] = (p3[1] - p4[1]) + (sc*(p2[1] - p3[1])) - (tc*(p6[1] - p4[1]));
dP[2] = (p3[2] - p4[2]) + (sc*(p2[2] - p3[2])) - (tc*(p6[2] - p4[2]));
}else if(j==6)
{
dP[0] = (p1[0] - p5[0]) + (sc*(p3[0] - p1[0])) - (tc*(p4[0] - p5[0]));
dP[1] = (p1[1] - p5[1]) + (sc*(p3[1] - p1[1])) - (tc*(p4[1] - p5[1]));
dP[2] = (p1[2] - p5[2]) + (sc*(p3[2] - p1[2])) - (tc*(p4[2] - p5[2]));
}else if(j==7)
{
dP[0] = (p1[0] - p6[0]) + (sc*(p3[0] - p1[0])) - (tc*(p5[0] - p6[0]));
dP[1] = (p1[1] - p6[1]) + (sc*(p3[1] - p1[1])) - (tc*(p5[1] - p6[1]));
dP[2] = (p1[2] - p6[2]) + (sc*(p3[2] - p1[2])) - (tc*(p5[2] - p6[2]));
}else
{
dP[0] = (p1[0] - p4[0]) + (sc*(p3[0] - p1[0])) - (tc*(p6[0] - p4[0]));
dP[1] = (p1[1] - p4[1]) + (sc*(p3[1] - p1[1])) - (tc*(p6[1] - p4[1]));
dP[2] = (p1[2] - p4[2]) + (sc*(p3[2] - p1[2])) - (tc*(p6[2] - p4[2]));
}

dist = sqrt(dP[0]*dP[0]+dP[1]*dP[1]+dP[2]*dP[2]);

if(dist<ssmin) {
ssmin = dist;
ttc = tc;
ssc = sc;
ssid = j;
}
}

switch(ssid)
{
case 0:
u[0] = (p1[0] - p2[0]);
u[1] = (p1[1] - p2[1]);
u[2] = (p1[2] - p2[2]);

v[0] = (p4[0] - p5[0]);
v[1] = (p4[1] - p5[1]);
v[2] = (p4[2] - p5[2]);

xPA = p2[0] + (u[0] * ssc);
yPA = p2[1] + (u[1] * ssc);
zPA = p2[2] + (u[2] * ssc);

xPB = p5[0] + (v[0] * ttc);
yPB = p5[1] + (v[1] * ttc);
zPB = p5[2] + (v[2] * ttc);
break;
case 1:
u[0] = (p1[0] - p2[0]);
u[1] = (p1[1] - p2[1]);
u[2] = (p1[2] - p2[2]);

v[0] = (p5[0] - p6[0]);
v[1] = (p5[1] - p6[1]);
v[2] = (p5[2] - p6[2]);

xPA = p2[0] + (u[0] * ssc);
yPA = p2[1] + (u[1] * ssc);
zPA = p2[2] + (u[2] * ssc);

xPB = p6[0] + (v[0] * ttc);
yPB = p6[1] + (v[1] * ttc);
zPB = p6[2] + (v[2] * ttc);
break;
case 2:
u[0] = (p1[0] - p2[0]);
u[1] = (p1[1] - p2[1]);
u[2] = (p1[2] - p2[2]);

v[0] = (p6[0] - p4[0]);
v[1] = (p6[1] - p4[1]);
v[2] = (p6[2] - p4[2]);

xPA = p2[0] + (u[0] * ssc);
yPA = p2[1] + (u[1] * ssc);
zPA = p2[2] + (u[2] * ssc);

xPB = p4[0] + (v[0] * ttc);
yPB = p4[1] + (v[1] * ttc);
zPB = p4[2] + (v[2] * ttc);
break;
case 3:
u[0] = (p2[0] - p3[0]);
u[1] = (p2[1] - p3[1]);
u[2] = (p2[2] - p3[2]);

v[0] = (p4[0] - p5[0]);
v[1] = (p4[1] - p5[1]);
v[2] = (p4[2] - p5[2]);

xPA = p3[0] + (u[0] * ssc);
yPA = p3[1] + (u[1] * ssc);
zPA = p3[2] + (u[2] * ssc);

xPB = p5[0] + (v[0] * ttc);
yPB = p5[1] + (v[1] * ttc);
zPB = p5[2] + (v[2] * ttc);
break;
case 4:
u[0] = (p2[0] - p3[0]);
u[1] = (p2[1] - p3[1]);
u[2] = (p2[2] - p3[2]);

v[0] = (p5[0] - p6[0]);
v[1] = (p5[1] - p6[1]);
v[2] = (p5[2] - p6[2]);

xPA = p3[0] + (u[0] * ssc);
yPA = p3[1] + (u[1] * ssc);
zPA = p3[2] + (u[2] * ssc);

xPB = p6[0] + (v[0] * ttc);
yPB = p6[1] + (v[1] * ttc);
zPB = p6[2] + (v[2] * ttc);
break;
case 5:
u[0] = (p2[0] - p3[0]);
u[1] = (p2[1] - p3[1]);
u[2] = (p2[2] - p3[2]);

v[0] = (p6[0] - p4[0]);
v[1] = (p6[1] - p4[1]);
v[2] = (p6[2] - p4[2]);

xPA = p3[0] + (u[0] * ssc);
yPA = p3[1] + (u[1] * ssc);
zPA = p3[2] + (u[2] * ssc);

xPB = p4[0] + (v[0] * ttc);
yPB = p4[1] + (v[1] * ttc);
zPB = p4[2] + (v[2] * ttc);
break;
case 6:
u[0] = (p3[0] - p1[0]);
u[1] = (p3[1] - p1[1]);
u[2] = (p3[2] - p1[2]);

v[0] = (p4[0] - p5[0]);
v[1] = (p4[1] - p5[1]);
v[2] = (p4[2] - p5[2]);

xPA = p1[0] + (u[0] * ssc);
yPA = p1[1] + (u[1] * ssc);
zPA = p1[2] + (u[2] * ssc);

xPB = p5[0] + (v[0] * ttc);
yPB = p5[1] + (v[1] * ttc);
zPB = p5[2] + (v[2] * ttc);
break;
case 7:
u[0] = (p3[0] - p1[0]);
u[1] = (p3[1] - p1[1]);
u[2] = (p3[2] - p1[2]);

v[0] = (p5[0] - p6[0]);
v[1] = (p5[1] - p6[1]);
v[2] = (p5[2] - p6[2]);

xPA = p1[0] + (u[0] * ssc);
yPA = p1[1] + (u[1] * ssc);
zPA = p1[2] + (u[2] * ssc);

xPB = p6[0] + (v[0] * ttc);
yPB = p6[1] + (v[1] * ttc);
zPB = p6[2] + (v[2] * ttc);
break;
case 8:
u[0] = (p3[0] - p1[0]);
u[1] = (p3[1] - p1[1]);
u[2] = (p3[2] - p1[2]);

v[0] = (p6[0] - p4[0]);
v[1] = (p6[1] - p4[1]);
v[2] = (p6[2] - p4[2]);

xPA = p1[0] + (u[0] * ssc);
yPA = p1[1] + (u[1] * ssc);
zPA = p1[2] + (u[2] * ssc);

xPB = p4[0] + (v[0] * ttc);
yPB = p4[1] + (v[1] * ttc);
zPB = p4[2] + (v[2] * ttc);
break;
}

a[0] = (p2[0] - p1[0])*(p2[0] - p1[0]) + (p2[1] - p1[1])*(p2[1] - p1[1]) + (p2[2] - p1[2])*(p2[2] - p1[2]);
b[0] = (p2[0] - p1[0])*(p3[0] - p1[0]) + (p2[1] - p1[1])*(p3[1] - p1[1]) + (p2[2] - p1[2])*(p3[2] - p1[2]);
c[0] = (p3[0] - p1[0])*(p3[0] - p1[0]) + (p3[1] - p1[1])*(p3[1] - p1[1]) + (p3[2] - p1[2])*(p3[2] - p1[2]);
d[0] = (p2[0] - p1[0])*(p1[0] - p4[0]) + (p2[1] - p1[1])*(p1[1] - p4[1]) + (p2[2] - p1[2])*(p1[2] - p4[2]);
e[0] = (p3[0] - p1[0])*(p1[0] - p4[0]) + (p3[1] - p1[1])*(p1[1] - p4[1]) + (p3[2] - p1[2])*(p1[2] - p4[2]);
f[0] = (p1[0] - p4[0])*(p1[0] - p4[0]) + (p1[1] - p4[1])*(p1[1] - p4[1]) + (p1[2] - p4[2])*(p1[2] - p4[2]);

a[1] = (p2[0] - p1[0])*(p2[0] - p1[0]) + (p2[1] - p1[1])*(p2[1] - p1[1]) + (p2[2] - p1[2])*(p2[2] - p1[2]);
b[1] = (p2[0] - p1[0])*(p3[0] - p1[0]) + (p2[1] - p1[1])*(p3[1] - p1[1]) + (p2[2] - p1[2])*(p3[2] - p1[2]);
c[1] = (p3[0] - p1[0])*(p3[0] - p1[0]) + (p3[1] - p1[1])*(p3[1] - p1[1]) + (p3[2] - p1[2])*(p3[2] - p1[2]);
d[1] = (p2[0] - p1[0])*(p1[0] - p5[0]) + (p2[1] - p1[1])*(p1[1] - p5[1]) + (p2[2] - p1[2])*(p1[2] - p5[2]);
e[1] = (p3[0] - p1[0])*(p1[0] - p5[0]) + (p3[1] - p1[1])*(p1[1] - p5[1]) + (p3[2] - p1[2])*(p1[2] - p5[2]);
f[1] = (p1[0] - p5[0])*(p1[0] - p5[0]) + (p1[1] - p5[1])*(p1[1] - p5[1]) + (p1[2] - p5[2])*(p1[2] - p5[2]);

a[2] = (p2[0] - p1[0])*(p2[0] - p1[0]) + (p2[1] - p1[1])*(p2[1] - p1[1]) + (p2[2] - p1[2])*(p2[2] - p1[2]);
b[2] = (p2[0] - p1[0])*(p3[0] - p1[0]) + (p2[1] - p1[1])*(p3[1] - p1[1]) + (p2[2] - p1[2])*(p3[2] - p1[2]);
c[2] = (p3[0] - p1[0])*(p3[0] - p1[0]) + (p3[1] - p1[1])*(p3[1] - p1[1]) + (p3[2] - p1[2])*(p3[2] - p1[2]);
d[2] = (p2[0] - p1[0])*(p1[0] - p6[0]) + (p2[1] - p1[1])*(p1[1] - p6[1]) + (p2[2] - p1[2])*(p1[2] - p6[2]);
e[2] = (p3[0] - p1[0])*(p1[0] - p6[0]) + (p3[1] - p1[1])*(p1[1] - p6[1]) + (p3[2] - p1[2])*(p1[2] - p6[2]);
f[2] = (p1[0] - p6[0])*(p1[0] - p6[0]) + (p1[1] - p6[1])*(p1[1] - p6[1]) + (p1[2] - p6[2])*(p1[2] - p6[2]);


a[3] = (p5[0] - p4[0])*(p5[0] - p4[0]) + (p5[1] - p4[1])*(p5[1] - p4[1]) + (p5[2] - p4[2])*(p5[2] - p4[2]);
b[3] = (p5[0] - p4[0])*(p6[0] - p4[0]) + (p5[1] - p4[1])*(p6[1] - p4[1]) + (p5[2] - p4[2])*(p6[2] - p4[2]);
c[3] = (p6[0] - p4[0])*(p6[0] - p4[0]) + (p6[1] - p4[1])*(p6[1] - p4[1]) + (p6[2] - p4[2])*(p6[2] - p4[2]);
d[3] = (p5[0] - p4[0])*(p4[0] - p1[0]) + (p5[1] - p4[1])*(p4[1] - p1[1]) + (p5[2] - p4[2])*(p4[2] - p1[2]);
e[3] = (p6[0] - p4[0])*(p4[0] - p1[0]) + (p6[1] - p4[1])*(p4[1] - p1[1]) + (p6[2] - p4[2])*(p4[2] - p1[2]);
f[3] = (p4[0] - p1[0])*(p4[0] - p1[0]) + (p4[1] - p1[1])*(p4[1] - p1[1]) + (p4[2] - p1[2])*(p4[2] - p1[2]);

a[4] = (p5[0] - p4[0])*(p5[0] - p4[0]) + (p5[1] - p4[1])*(p5[1] - p4[1]) + (p5[2] - p4[2])*(p5[2] - p4[2]);
b[4] = (p5[0] - p4[0])*(p6[0] - p4[0]) + (p5[1] - p4[1])*(p6[1] - p4[1]) + (p5[2] - p4[2])*(p6[2] - p4[2]);
c[4] = (p6[0] - p4[0])*(p6[0] - p4[0]) + (p6[1] - p4[1])*(p6[1] - p4[1]) + (p6[2] - p4[2])*(p6[2] - p4[2]);
d[4] = (p5[0] - p4[0])*(p4[0] - p2[0]) + (p5[1] - p4[1])*(p4[1] - p2[1]) + (p5[2] - p4[2])*(p4[2] - p2[2]);
e[4] = (p6[0] - p4[0])*(p4[0] - p2[0]) + (p6[1] - p4[1])*(p4[1] - p2[1]) + (p6[2] - p4[2])*(p4[2] - p2[2]);
f[4] = (p4[0] - p2[0])*(p4[0] - p2[0]) + (p4[1] - p2[1])*(p4[1] - p2[1]) + (p4[2] - p2[2])*(p4[2] - p2[2]);

a[5] = (p5[0] - p4[0])*(p5[0] - p4[0]) + (p5[1] - p4[1])*(p5[1] - p4[1]) + (p5[2] - p4[2])*(p5[2] - p4[2]);
b[5] = (p5[0] - p4[0])*(p6[0] - p4[0]) + (p5[1] - p4[1])*(p6[1] - p4[1]) + (p5[2] - p4[2])*(p6[2] - p4[2]);
c[5] = (p6[0] - p4[0])*(p6[0] - p4[0]) + (p6[1] - p4[1])*(p6[1] - p4[1]) + (p6[2] - p4[2])*(p6[2] - p4[2]);
d[5] = (p5[0] - p4[0])*(p4[0] - p3[0]) + (p5[1] - p4[1])*(p4[1] - p3[1]) + (p5[2] - p4[2])*(p4[2] - p3[2]);
e[5] = (p6[0] - p4[0])*(p4[0] - p3[0]) + (p6[1] - p4[1])*(p4[1] - p3[1]) + (p6[2] - p4[2])*(p4[2] - p3[2]);
f[5] = (p4[0] - p3[0])*(p4[0] - p3[0]) + (p4[1] - p3[1])*(p4[1] - p3[1]) + (p4[2] - p3[2])*(p4[2] - p3[2]);

iREAL ss = 0.0, tt = 0.0;
iREAL ptmin = 1E30;
int id=0;
for( int j=0; j<6;j++)
{
iREAL det = a[j]*c[j] - b[j]*b[j];
iREAL s   = b[j]*e[j] - c[j]*d[j];
iREAL t   = b[j]*d[j] - a[j]*e[j];

iREAL sqrDistance=0.0;

if ((s+t) <= det)
{
if (s < 0){
if (t < 0){
if (d[j] < 0){
t = 0;
if (-d[j] >= a[j]){
s = 1;
sqrDistance = a[j] + 2*d[j] + f[j];
}else {
s = -d[j]/a[j];
sqrDistance = d[j]*s + f[j];
}
}else {
s = 0;
if (e[j] >= 0){
t = 0;
sqrDistance = f[j];
}else{
if (-e[j] >= c[j]){
t = 1;
sqrDistance = c[j] + 2*e[j] + f[j];
} else {
t = -e[j]/c[j];
sqrDistance = e[j]*t + f[j];
}
}
} 
}else {
s = 0;
if (e[j] >= 0){
t = 0;
sqrDistance = f[j];
}else {
if (-e[j] >= c[j]){
t = 1;
sqrDistance = c[j] + 2*e[j] +f[j];
}else {
t = -e[j]/c[j];
sqrDistance = e[j]*t + f[j];
}
}
} 
}else {
if (t < 0){
t = 0;
if (d[j] >= 0){
s = 0;
sqrDistance = f[j];
}else {
if (-d[j] >= a[j]){
s = 1;
sqrDistance = a[j] + 2*d[j] + f[j];
}else {
s = -d[j]/a[j];
sqrDistance = d[j]*s + f[j];
}
}
}else {
iREAL invDet = 1/det;
s = s*invDet;
t = t*invDet;
sqrDistance = s*(a[j]*s + b[j]*t + 2*d[j]) + t*(b[j]*s + c[j]*t + 2*e[j]) + f[j];
}
}
}
else
{
if (s < 0){
iREAL tmp0 = b[j] + d[j];
iREAL tmp1 = c[j] + e[j];
if (tmp1 > tmp0){ 
iREAL numer = tmp1 - tmp0;
iREAL denom = a[j] - 2*b[j] + c[j];
if (numer >= denom){
s = 1;
t = 0;
sqrDistance = a[j] + 2*d[j] + f[j]; 
}else {
s = numer/denom;
t = 1-s;
sqrDistance = s*(a[j]*s + b[j]*t + 2*d[j]) + t*(b[j]*s + c[j]*t + 2*e[j]) + f[j];
}
}else {         
s = 0;
if (tmp1 <= 0) {
t = 1;
sqrDistance = c[j] + 2*e[j] + f[j];
}else {
if (e[j] >= 0){
t = 0;
sqrDistance = f[j];
}else {
t = -e[j]/c[j];
sqrDistance = e[j]*t + f[j];
}
}
} 
}else {
if (t < 0) {
iREAL tmp0 = b[j] + e[j];
iREAL tmp1 = a[j] + d[j];
if (tmp1 > tmp0){
iREAL numer = tmp1 - tmp0;
iREAL denom = a[j]-2*b[j]+c[j];
if (numer >= denom){
t = 1;
s = 0;
sqrDistance = c[j] + 2*e[j] + f[j];
}else {
t = numer/denom;
s = 1 - t;
sqrDistance = s*(a[j]*s + b[j]*t + 2*d[j]) + t*(b[j]*s + c[j]*t + 2*e[j]) + f[j];
}
}else {
t = 0;
if (tmp1 <= 0){
s = 1;
sqrDistance = a[j] + 2*d[j] + f[j];
}else {
if (d[j] >= 0) {
s = 0;
sqrDistance = f[j];
}else {
s = -d[j]/a[j];
sqrDistance = d[j]*s + f[j];
}
}
}
}else {
iREAL numer = c[j] + e[j] - b[j] - d[j];
if (numer <= 0){
s = 0;
t = 1;
sqrDistance = c[j] + 2*e[j] + f[j];
}else {
iREAL denom = a[j] - 2*b[j] + c[j];
if (numer >= denom){
s = 1;
t = 0;
sqrDistance = a[j] + 2*d[j] + f[j];
}else {
s = numer/denom;
t = 1-s;
sqrDistance = s*(a[j]*s + b[j]*t + 2*d[j]) + t*(b[j]*s + c[j]*t + 2*e[j]) + f[j];
}
} 
}
}
}

if (sqrDistance <= 0.0){sqrDistance = 0.0;}

iREAL dist = sqrt(sqrDistance);
if(dist<ptmin)
{
ptmin = dist;
ss = s;
tt = t;
id = j;
}
}

if(ptmin > ssmin) {return;}

switch(id)
{
case 0:
xPA = p1[0] + ((p3[0] - p1[0]) * tt) + ((p2[0] - p1[0]) * ss);
yPA = p1[1] + ((p3[1] - p1[1]) * tt) + ((p2[1] - p1[1]) * ss);
zPA = p1[2] + ((p3[2] - p1[2]) * tt) + ((p2[2] - p1[2]) * ss);

xPB = p4[0];
yPB = p4[1];
zPB = p4[2];
break;
case 1:
xPA = p1[0] + ((p3[0] - p1[0]) * tt) + ((p2[0] - p1[0]) * ss);
yPA = p1[1] + ((p3[1] - p1[1]) * tt) + ((p2[1] - p1[1]) * ss);
zPA = p1[2] + ((p3[2] - p1[2]) * tt) + ((p2[2] - p1[2]) * ss);

xPB = p5[0];
yPB = p5[1];
zPB = p5[2];

break;
case 2:
xPA = p1[0] + ((p3[0] - p1[0]) * tt) + ((p2[0] - p1[0]) * ss);
yPA = p1[1] + ((p3[1] - p1[1]) * tt) + ((p2[1] - p1[1]) * ss);
zPA = p1[2] + ((p3[2] - p1[2]) * tt) + ((p2[2] - p1[2]) * ss);

xPB = p6[0];
yPB = p6[1];
zPB = p6[2];

break;
case 3:
xPA = p1[0];
yPA = p1[1];
zPA = p1[2];

xPB = p4[0] + ((p6[0] - p4[0]) * tt) + ((p5[0] - p4[0]) * ss);
yPB = p4[1] + ((p6[1] - p4[1]) * tt) + ((p5[1] - p4[1]) * ss);
zPB = p4[2] + ((p6[2] - p4[2]) * tt) + ((p5[2] - p4[2]) * ss);

break;
case 4:
xPA = p2[0];
yPA = p2[1];
zPA = p2[2];

xPB = p4[0] + ((p6[0] - p4[0]) * tt) + ((p5[0] - p4[0]) * ss);
yPB = p4[1] + ((p6[1] - p4[1]) * tt) + ((p5[1] - p4[1]) * ss);
zPB = p4[2] + ((p6[2] - p4[2]) * tt) + ((p5[2] - p4[2]) * ss);

break;
case 5:
xPA = p3[0];
yPA = p3[1];
zPA = p3[2];

xPB = p4[0] + ((p6[0] - p4[0]) * tt) + ((p5[0] - p4[0]) * ss);
yPB = p4[1] + ((p6[1] - p4[1]) * tt) + ((p5[1] - p4[1]) * ss);
zPB = p4[2] + ((p6[2] - p4[2]) * tt) + ((p5[2] - p4[2]) * ss);

break;
}
}
