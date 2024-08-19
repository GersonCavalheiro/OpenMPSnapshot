

#include "omp.h"
#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

#define NM (2+(1<<LM)) 
#define NV (ONE*(2+(1<<NDIM1))*(2+(1<<NDIM2))*(2+(1<<NDIM3))) 
#define NR (((NV+NM*NM+5*NM+7*LM+6)/7)*8) 
#define MAXLEVEL (LT_DEFAULT+1) 
#define M (NM+1) 
#define MM (10)
#define	A (pow(5.0,13.0))
#define	X (314159265.0)
#define T_INIT 0
#define T_BENCH 1
#define T_MG3P 2
#define T_PSINV 3
#define T_RESID 4
#define T_RESID2 5
#define T_RPRJ3 6
#define T_INTERP 7
#define T_NORM2 8
#define T_COMM3 9
#define T_LAST 10


#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
static int nx[MAXLEVEL+1];
static int ny[MAXLEVEL+1];
static int nz[MAXLEVEL+1];
static int m1[MAXLEVEL+1];
static int m2[MAXLEVEL+1];
static int m3[MAXLEVEL+1];
static int ir[MAXLEVEL+1];
static int debug_vec[8];
static double u[NR];
static double v[NV];
static double r[NR];
#else
static int (*nx)=(int*)malloc(sizeof(int)*(MAXLEVEL+1));
static int (*ny)=(int*)malloc(sizeof(int)*(MAXLEVEL+1));
static int (*nz)=(int*)malloc(sizeof(int)*(MAXLEVEL+1));
static int (*m1)=(int*)malloc(sizeof(int)*(MAXLEVEL+1));
static int (*m2)=(int*)malloc(sizeof(int)*(MAXLEVEL+1));
static int (*m3)=(int*)malloc(sizeof(int)*(MAXLEVEL+1));
static int (*ir)=(int*)malloc(sizeof(int)*(MAXLEVEL+1));
static int (*debug_vec)=(int*)malloc(sizeof(int)*(8));
static double (*u)=(double*)malloc(sizeof(double)*(NR));
static double (*v)=(double*)malloc(sizeof(double)*(NV));
static double (*r)=(double*)malloc(sizeof(double)*(NR));
#endif
static int is1, is2, is3, ie1, ie2, ie3, lt, lb;
static boolean timeron;


static void bubble(double ten[][MM], int j1[][MM], int j2[][MM], int j3[][MM], int m, int ind);
static void comm3(void* pointer_u, int n1, int n2, int n3, int kk);
static void interp(void* pointer_z, int mm1, int mm2, int mm3, void* pointer_u, int n1, int n2, int n3, int k);
static void mg3P(double u[], double v[], double r[], double a[4], double c[4], int n1, int n2, int n3, int k);
static void norm2u3(void* pointer_r, int n1, int n2, int n3, double* rnm2, double* rnmu, int nx, int ny, int nz);
static double power(double a, int n);
static void psinv(void* pointer_r, void* pointer_u, int n1, int n2, int n3, double c[4], int k);
static void rep_nrm(void* pointer_u, int n1, int n2, int n3, char* title, int kk);
static void resid(void* pointer_u, void* pointer_v, void* pointer_r, int n1, int n2, int n3, double a[4], int k);
static void rprj3(void* pointer_r, int m1k, int m2k, int m3k, void* pointer_s, int m1j, int m2j, int m3j, int k);
static void setup(int* n1, int* n2, int* n3, int k);
static void showall(void* pointer_z, int n1, int n2, int n3);
static void zero3(void* pointer_z, int n1, int n2, int n3);
static void zran3(void* pointer_z, int n1, int n2, int n3, int nx, int ny, int k);


int main(int argc, char *argv[]){
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
#endif

int k, it;
double t, tinit, mflops;

double a[4], c[4];

double rnm2, rnmu, epsilon;
int n1, n2, n3, nit;
double nn, verify_value, err;
boolean verified;
char class_npb;

int i;
char* t_names[T_LAST];
double tmax;

for(i=T_INIT; i<T_LAST; i++){
timer_clear(i);
}

timer_start(T_INIT);	


FILE* fp;
if((fp = fopen("timer.flag", "r")) != NULL){
timeron = TRUE;
t_names[T_INIT] = (char*) "init";
t_names[T_BENCH] = (char*) "benchmk";
t_names[T_MG3P] = (char*) "mg3P";
t_names[T_PSINV] = (char*) "psinv";
t_names[T_RESID] = (char*) "resid";
t_names[T_RPRJ3] = (char*) "rprj3";
t_names[T_INTERP] = (char*) "interp";
t_names[T_NORM2] = (char*) "norm2";
t_names[T_COMM3] = (char*) "comm3";
fclose(fp);
}else{
timeron = FALSE;
}
fp = fopen("mg.input", "r");
if(fp != NULL){
printf(" Reading from input file mg.input\n");
if(fscanf(fp, "%d", &lt) != 1){
printf(" Error in reading elements\n");
exit(1);
}
while(fgetc(fp) != '\n');
if(fscanf(fp, "%d%d%d", &nx[lt], &ny[lt], &nz[lt]) != 3){
printf(" Error in reading elements\n");
exit(1);
}
while(fgetc(fp) != '\n');
if(fscanf(fp, "%d", &nit) != 1){
printf(" Error in reading elements\n");
exit(1);
}
while(fgetc(fp) != '\n');
for(i = 0; i <= 7; i++) {
if(fscanf(fp, "%d", &debug_vec[i]) != 1){
printf(" Error in reading elements\n");
exit(1);
}
}
fclose(fp);
}else{
printf(" No input file. Using compiled defaults\n");
lt = LT_DEFAULT;
nit = NIT_DEFAULT;
nx[lt] = NX_DEFAULT;
ny[lt] = NY_DEFAULT;
nz[lt] = NZ_DEFAULT;
for(i = 0; i <= 7; i++){
debug_vec[i] = DEBUG_DEFAULT;
}
}

if((nx[lt] != ny[lt]) || (nx[lt] != nz[lt])){
class_npb = 'U';
}else if(nx[lt] == 32 && nit == 4){
class_npb = 'S';
}else if(nx[lt] == 128 && nit == 4){
class_npb = 'W';
}else if(nx[lt] == 256 && nit == 4){
class_npb = 'A';
}else if(nx[lt] == 256 && nit == 20){
class_npb = 'B';
}else if(nx[lt] == 512 && nit == 20){
class_npb = 'C';
}else if(nx[lt] == 1024 && nit == 50){  
class_npb = 'D';
}else if(nx[lt] == 2048 && nit == 50){  
class_npb = 'E';	
}else{
class_npb = 'U';
}


a[0] = -8.0/3.0;
a[1] =  0.0;
a[2] =  1.0/6.0;
a[3] =  1.0/12.0;

if(class_npb == 'A' || class_npb == 'S' || class_npb =='W'){

c[0] =  -3.0/8.0;
c[1] =  +1.0/32.0;
c[2] =  -1.0/64.0;
c[3] =   0.0;
}else{

c[0] =  -3.0/17.0;
c[1] =  +1.0/33.0;
c[2] =  -1.0/61.0;
c[3] =   0.0;
}

lb = 1;
k = lt;

setup(&n1,&n2,&n3,k);

zero3(u,n1,n2,n3);

zran3(v,n1,n2,n3,nx[lt],ny[lt],k);

norm2u3(v,n1,n2,n3,&rnm2,&rnmu,nx[lt],ny[lt],nz[lt]);

printf("\n\n NAS Parallel Benchmarks 4.1 Parallel C++ version with OpenMP - MG Benchmark\n\n");
printf(" Size: %3dx%3dx%3d (class_npb %1c)\n", nx[lt], ny[lt], nz[lt], class_npb);
printf(" Iterations: %3d\n", nit);

#pragma omp parallel
{
resid(u,v,r,n1,n2,n3,a,k);

norm2u3(r,n1,n2,n3,&rnm2,&rnmu,nx[lt],ny[lt],nz[lt]);


mg3P(u,v,r,a,c,n1,n2,n3,k);
resid(u,v,r,n1,n2,n3,a,k);
}

setup(&n1,&n2,&n3,k);

zero3(u,n1,n2,n3);

zran3(v,n1,n2,n3,nx[lt],ny[lt],k);

timer_stop(T_INIT);
tinit = timer_read(T_INIT);
printf(" Initialization time: %15.3f seconds\n", tinit);

for(i=T_BENCH; i<T_LAST; i++){
timer_clear(i);
}
timer_start(T_BENCH);

#pragma omp parallel firstprivate(nit) private(it)
{

if(timeron){
#pragma omp master
timer_start(T_RESID2);
}

resid(u,v,r,n1,n2,n3,a,k);

if(timeron){
#pragma omp master
timer_stop(T_RESID2);
}

norm2u3(r,n1,n2,n3,&rnm2,&rnmu,nx[lt],ny[lt],nz[lt]);

for(it = 1; it <= nit; it++){
if((it==1)||(it==nit)||((it%5)==0)){
#pragma omp master
printf("  iter %3d\n",it);
}

if(timeron){
#pragma omp master
timer_start(T_MG3P);
}

mg3P(u,v,r,a,c,n1,n2,n3,k);

if(timeron){
#pragma omp master
timer_stop(T_MG3P);
}
if(timeron){
#pragma omp master
timer_start(T_RESID2);
}

resid(u,v,r,n1,n2,n3,a,k);

if(timeron){
#pragma omp master
timer_stop(T_RESID2);
}
}

norm2u3(r,n1,n2,n3,&rnm2,&rnmu,nx[lt],ny[lt],nz[lt]);
} 

timer_stop(T_BENCH);

t = timer_read(T_BENCH);    	

verified = FALSE;
verify_value = 0.0;	

printf(" Benchmark completed\n");

epsilon = 1.0e-8;
if(class_npb != 'U'){
if(class_npb == 'S'){
verify_value = 0.5307707005734e-04;
}else if(class_npb == 'W'){
verify_value = 0.6467329375339e-05;
}else if(class_npb == 'A'){
verify_value = 0.2433365309069e-05;
}else if(class_npb == 'B'){
verify_value = 0.1800564401355e-05;
}else if(class_npb == 'C'){
verify_value = 0.5706732285740e-06;
}else if(class_npb == 'D'){
verify_value = 0.1583275060440e-09;
}else if(class_npb == 'E'){
verify_value = 0.8157592357404e-10; 
}

err = fabs(rnm2-verify_value) / verify_value;
if(err <= epsilon){
verified = TRUE;
printf(" VERIFICATION SUCCESSFUL\n");
printf(" L2 Norm is %20.13e\n", rnm2);
printf(" Error is   %20.13e\n", err);
}else{
verified = FALSE;
printf(" VERIFICATION FAILED\n");
printf(" L2 Norm is             %20.13e\n", rnm2);
printf(" The correct L2 Norm is %20.13e\n", verify_value);
}
}else{
verified = FALSE;
printf(" Problem size unknown\n");
printf(" NO VERIFICATION PERFORMED\n");
}

nn = 1.0*nx[lt]*ny[lt]*nz[lt];

if(t!=0.0){
mflops = 58.0*nit*nn*1.0e-6/t;
}else{
mflops = 0.0;
}

setenv("OMP_NUM_THREADS","1",0);
c_print_results((char*)"MG",
class_npb,
nx[lt],
ny[lt],
nz[lt],
nit,
t,
mflops,
(char*)"          floating point",
verified,
(char*)NPBVERSION,
(char*)COMPILETIME,
(char*)COMPILERVERSION,
(char*)LIBVERSION,
std::getenv("OMP_NUM_THREADS"),
(char*)CS1,
(char*)CS2,
(char*)CS3,
(char*)CS4,
(char*)CS5,
(char*)CS6,
(char*)CS7);


if(timeron){
tmax = timer_read(T_BENCH);
if(tmax==0.0){tmax=1.0;}
printf("  SECTION   Time (secs)\n");
for(i=T_BENCH; i<T_LAST; i++){
t = timer_read(i);
if(i==T_RESID2){
t = timer_read(T_RESID) - t;
printf("    --> %8s:%9.3f  (%6.2f%%)\n", "mg-resid", t, t*100.0/tmax);
}else{
printf("  %-8s:%9.3f  (%6.2f%%)\n", t_names[i], t, t*100.0/tmax);
}
}
}

return 0;
}


static void bubble(double ten[][MM], int j1[][MM], int j2[][MM], int j3[][MM], int m, int ind){
double temp;
int i, j_temp;

if(ind == 1){
for(i = 0; i < m-1; i++){
if(ten[ind][i] > ten[ind][i+1]){
temp = ten[ind][i+1];
ten[ind][i+1] = ten[ind][i];
ten[ind][i] = temp;

j_temp = j1[ind][i+1];
j1[ind][i+1] = j1[ind][i];
j1[ind][i] = j_temp;

j_temp = j2[ind][i+1];
j2[ind][i+1] = j2[ind][i];
j2[ind][i] = j_temp;

j_temp = j3[ind][i+1];
j3[ind][i+1] = j3[ind][i];
j3[ind][i] = j_temp;
}else{
return;
}
}
}else{
for(i = 0; i < m-1; i++){
if(ten[ind][i] < ten[ind][i+1]){
temp = ten[ind][i+1];
ten[ind][i+1] = ten[ind][i];
ten[ind][i] = temp;

j_temp = j1[ind][i+1];
j1[ind][i+1] = j1[ind][i];
j1[ind][i] = j_temp;

j_temp = j2[ind][i+1];
j2[ind][i+1] = j2[ind][i];
j2[ind][i] = j_temp;

j_temp = j3[ind][i+1];
j3[ind][i+1] = j3[ind][i];
j3[ind][i] = j_temp;
}else{
return;
}
}
}
}


static void comm3(void* pointer_u, int n1, int n2, int n3, int kk){
#ifdef __clang__
using custom_cast = double (*)[n2][n1];
custom_cast u = reinterpret_cast<custom_cast>(pointer_u);
#else
double (*u)[n2][n1] = (double (*)[n2][n1])pointer_u;
#endif

int i1, i2, i3;
if(timeron){
#pragma omp master
timer_start(T_COMM3);
}
#pragma omp for
for(i3 = 1; i3 < n3-1; i3++){

for(i2 = 1; i2 < n2-1; i2++){
u[i3][i2][0] = u[i3][i2][n1-2];
u[i3][i2][n1-1] = u[i3][i2][1];			
}

for(i1 = 0; i1 < n1; i1++){
u[i3][0][i1] = u[i3][n2-2][i1];
u[i3][n2-1][i1] = u[i3][1][i1];			
}
}

#pragma omp for
for(i2 = 0; i2 < n2; i2++){
for(i1 = 0; i1 < n1; i1++){
u[0][i2][i1] = u[n3-2][i2][i1];
u[n3-1][i2][i1] = u[1][i2][i1];			
}
}

if(timeron){
#pragma omp master
timer_stop(T_COMM3);
}
}


static void interp(void* pointer_z, int mm1, int mm2, int mm3, void* pointer_u, int n1, int n2, int n3, int k){
#ifdef __clang__
using custom_cast = double (*)[mm2][mm1];
custom_cast z = reinterpret_cast<custom_cast>(pointer_z);
using custom_cast2 = double (*)[n2][n1];
custom_cast2 u = reinterpret_cast<custom_cast2>(pointer_u);
#else
double (*z)[mm2][mm1] = (double (*)[mm2][mm1])pointer_z;
double (*u)[n2][n1] = (double (*)[n2][n1])pointer_u;
#endif

int i3, i2, i1, d1, d2, d3, t1, t2, t3;


double z1[M], z2[M], z3[M];

if(timeron){
#pragma omp master
timer_start(T_INTERP);
}

if(n1 != 3 && n2 != 3 && n3 != 3){
#pragma omp for
for(i3 = 0; i3 < mm3-1; i3++){
for(i2 = 0; i2 < mm2-1; i2++){
for(i1 = 0; i1 < mm1; i1++){
z1[i1] = z[i3][i2+1][i1] + z[i3][i2][i1];
z2[i1] = z[i3+1][i2][i1] + z[i3][i2][i1];
z3[i1] = z[i3+1][i2+1][i1] + z[i3+1][i2][i1] + z1[i1];
}
for(i1 = 0; i1 < mm1-1; i1++){
u[2*i3][2*i2][2*i1] = u[2*i3][2*i2][2*i1]
+z[i3][i2][i1];
u[2*i3][2*i2][2*i1+1] = u[2*i3][2*i2][2*i1+1]
+0.5*(z[i3][i2][i1+1]+z[i3][i2][i1]);
}
for(i1 = 0; i1 < mm1-1; i1++){
u[2*i3][2*i2+1][2*i1] = u[2*i3][2*i2+1][2*i1]
+0.5 * z1[i1];
u[2*i3][2*i2+1][2*i1+1] = u[2*i3][2*i2+1][2*i1+1]
+0.25*( z1[i1] + z1[i1+1] );
}
for(i1 = 0; i1 < mm1-1; i1++){
u[2*i3+1][2*i2][2*i1] = u[2*i3+1][2*i2][2*i1]
+0.5 * z2[i1];
u[2*i3+1][2*i2][2*i1+1] = u[2*i3+1][2*i2][2*i1+1]
+0.25*( z2[i1] + z2[i1+1] );
}
for(i1 = 0; i1 < mm1-1; i1++){
u[2*i3+1][2*i2+1][2*i1] = u[2*i3+1][2*i2+1][2*i1]
+0.25* z3[i1];
u[2*i3+1][2*i2+1][2*i1+1] = u[2*i3+1][2*i2+1][2*i1+1]
+0.125*( z3[i1] + z3[i1+1] );
}
}
}
}else{
if(n1 == 3){
d1 = 2;
t1 = 1;
}else{
d1 = 1;
t1 = 0;
}      
if(n2 == 3){
d2 = 2;
t2 = 1;
}else{
d2 = 1;
t2 = 0;
}          
if(n3 == 3){
d3 = 2;
t3 = 1;
}else{
d3 = 1;
t3 = 0;
}
#pragma omp for
for(i3 = d3; i3 <= mm3-1; i3++){
for(i2 = d2; i2 <= mm2-1; i2++){
for(i1 = d1; i1 <= mm1-1; i1++){
u[2*i3-d3-1][2*i2-d2-1][2*i1-d1-1] =
u[2*i3-d3-1][2*i2-d2-1][2*i1-d1-1]
+z[i3-1][i2-1][i1-1];
}
for(i1 = 1; i1 <= mm1-1; i1++){
u[2*i3-d3-1][2*i2-d2-1][2*i1-t1-1] =
u[2*i3-d3-1][2*i2-d2-1][2*i1-t1-1]
+0.5*(z[i3-1][i2-1][i1]+z[i3-1][i2-1][i1-1]);
}
}
for(i2 = 1; i2 <= mm2-1; i2++){
for ( i1 = d1; i1 <= mm1-1; i1++) {
u[2*i3-d3-1][2*i2-t2-1][2*i1-d1-1] =
u[2*i3-d3-1][2*i2-t2-1][2*i1-d1-1]
+0.5*(z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1]);
}
for(i1 = 1; i1 <= mm1-1; i1++){
u[2*i3-d3-1][2*i2-t2-1][2*i1-t1-1] =
u[2*i3-d3-1][2*i2-t2-1][2*i1-t1-1]
+0.25*(z[i3-1][i2][i1]+z[i3-1][i2-1][i1]
+z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1]);
}
}
}
#pragma omp for
for(i3 = 1; i3 <= mm3-1; i3++){
for(i2 = d2; i2 <= mm2-1; i2++){
for(i1 = d1; i1 <= mm1-1; i1++){
u[2*i3-t3-1][2*i2-d2-1][2*i1-d1-1] =
u[2*i3-t3-1][2*i2-d2-1][2*i1-d1-1]
+0.5*(z[i3][i2-1][i1-1]+z[i3-1][i2-1][i1-1]);
}
for(i1 = 1; i1 <= mm1-1; i1++){
u[2*i3-t3-1][2*i2-d2-1][2*i1-t1-1] =
u[2*i3-t3-1][2*i2-d2-1][2*i1-t1-1]
+0.25*(z[i3][i2-1][i1]+z[i3][i2-1][i1-1]
+z[i3-1][i2-1][i1]+z[i3-1][i2-1][i1-1]);
}
}
for(i2 = 1; i2 <= mm2-1; i2++){
for (i1 = d1; i1 <= mm1-1; i1++){
u[2*i3-t3-1][2*i2-t2-1][2*i1-d1-1] =
u[2*i3-t3-1][2*i2-t2-1][2*i1-d1-1]
+0.25*(z[i3][i2][i1-1]+z[i3][i2-1][i1-1]
+z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1]);
}
for(i1 = 1; i1 <= mm1-1; i1++){
u[2*i3-t3-1][2*i2-t2-1][2*i1-t1-1] =
u[2*i3-t3-1][2*i2-t2-1][2*i1-t1-1]
+0.125*(z[i3][i2][i1]+z[i3][i2-1][i1]
+z[i3][i2][i1-1]+z[i3][i2-1][i1-1]
+z[i3-1][i2][i1]+z[i3-1][i2-1][i1]
+z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1]);
}
}
}
}
if(timeron){
#pragma omp master
timer_stop(T_INTERP);
}
#pragma omp single
{
if(debug_vec[0] >= 1){
rep_nrm(z,mm1,mm2,mm3,(char*)"z: inter",k-1);
rep_nrm(u,n1,n2,n3,(char*)"u: inter",k);
}
if(debug_vec[5] >= k){
showall(z,mm1,mm2,mm3);
showall(u,n1,n2,n3);
}
}
}


static void mg3P(double u[], double v[], double r[], double a[4], double c[4], int n1, int n2, int n3, int k){
int j;


for(k = lt; k >= lb+1; k--){
j = k-1;
rprj3(&r[ir[k]], m1[k], m2[k], m3[k], &r[ir[j]], m1[j], m2[j], m3[j], k);
}

k = lb;

zero3(&u[ir[k]], m1[k], m2[k], m3[k]);
psinv(&r[ir[k]], &u[ir[k]], m1[k], m2[k], m3[k], c, k);

for(k = lb+1; k <= lt-1; k++){
j = k-1;

zero3(&u[ir[k]], m1[k], m2[k], m3[k]);
interp(&u[ir[j]], m1[j], m2[j], m3[j], &u[ir[k]], m1[k], m2[k], m3[k], k);

resid(&u[ir[k]], &r[ir[k]], &r[ir[k]], m1[k], m2[k], m3[k], a, k);	

psinv(&r[ir[k]], &u[ir[k]], m1[k], m2[k], m3[k], c, k);
}

j = lt - 1;
k = lt;
interp(&u[ir[j]], m1[j], m2[j], m3[j], u, n1, n2, n3, k);	
resid(u, v, r, n1, n2, n3, a, k);	
psinv(r, u, n1, n2, n3, c, k);
}


static void norm2u3(void* pointer_r, int n1, int n2, int n3, double* rnm2, double* rnmu, int nx, int ny, int nz){
#ifdef __clang__
using custom_cast = double (*)[n2][n1];
custom_cast r = reinterpret_cast<custom_cast>(pointer_r);
#else
double (*r)[n2][n1] = (double (*)[n2][n1])pointer_r;
#endif	

static double s, rnmu_local;
double a;
int i3, i2, i1;

double dn;

if(timeron){
#pragma omp master
timer_start(T_NORM2);
}
dn = 1.0*nx*ny*nz;

#pragma omp single
{
s = 0.0;
rnmu_local = 0.0;
}

#pragma omp for reduction(+:s,rnmu_local) 
for(i3 = 1; i3 < n3-1; i3++){
for(i2 = 1; i2 < n2-1; i2++){
for(i1 = 1; i1 < n1-1; i1++){
s = s + r[i3][i2][i1] * r[i3][i2][i1];
a = fabs(r[i3][i2][i1]);
if(a > rnmu_local){rnmu_local = a;}
}
}
}

*rnmu = rnmu_local;
*rnm2 = sqrt(s/dn);
if(timeron){
#pragma omp master
timer_stop(T_NORM2);
}
}


static double power(double a, int n){
double aj;
int nj;
double rdummy;
double power;

power = 1.0;
nj = n;
aj = a;

while(nj != 0){
if((nj%2)==1){rdummy = randlc(&power, aj);}
rdummy = randlc(&aj, aj);
nj = nj/2;
}

return power;
}


static void psinv(void* pointer_r, void* pointer_u, int n1, int n2, int n3, double c[4], int k){
#ifdef __clang__
using custom_cast = double (*)[n2][n1];
custom_cast r = reinterpret_cast<custom_cast>(pointer_r);	
using custom_cast2 = double (*)[n2][n1];
custom_cast2 u = reinterpret_cast<custom_cast2>(pointer_u);
#else
double (*r)[n2][n1] = (double (*)[n2][n1])pointer_r;
double (*u)[n2][n1] = (double (*)[n2][n1])pointer_u;	
#endif		

int i3, i2, i1;
double r1[M], r2[M];

if(timeron){
#pragma omp master
timer_start(T_PSINV);
}

#pragma omp for
for(i3 = 1; i3 < n3-1; i3++){
for(i2 = 1; i2 < n2-1; i2++){
for(i1 = 0; i1 < n1; i1++){
r1[i1] = r[i3][i2-1][i1] + r[i3][i2+1][i1]
+ r[i3-1][i2][i1] + r[i3+1][i2][i1];
r2[i1] = r[i3-1][i2-1][i1] + r[i3-1][i2+1][i1]
+ r[i3+1][i2-1][i1] + r[i3+1][i2+1][i1];
}
for(i1 = 1; i1 < n1-1; i1++){
u[i3][i2][i1] = u[i3][i2][i1]
+ c[0] * r[i3][i2][i1]
+ c[1] * ( r[i3][i2][i1-1] + r[i3][i2][i1+1]
+ r1[i1] )
+ c[2] * ( r2[i1] + r1[i1-1] + r1[i1+1] );

}
}
}
if(timeron){
#pragma omp master
timer_stop(T_PSINV);
}


comm3(u,n1,n2,n3,k);

if(debug_vec[0] >= 1){
#pragma omp single
rep_nrm(u,n1,n2,n3,(char*)"   psinv",k);
}

if(debug_vec[3] >= k){
#pragma omp single
showall(u,n1,n2,n3);
}
}


static void rep_nrm(void* pointer_u, int n1, int n2, int n3, char* title, int kk){
double rnm2, rnmu;
norm2u3(pointer_u,n1,n2,n3,&rnm2,&rnmu,nx[kk],ny[kk],nz[kk]);
#pragma omp master
printf(" Level%2d in %8s: norms =%21.14e%21.14e\n", kk, title, rnm2, rnmu);
}


static void resid(void* pointer_u, void* pointer_v, void* pointer_r, int n1, int n2, int n3, double a[4], int k){
#ifdef __clang__
using custom_cast = double (*)[n2][n1];
custom_cast u = reinterpret_cast<custom_cast>(pointer_u);	
using custom_cast2 = double (*)[n2][n1];
custom_cast2 v = reinterpret_cast<custom_cast2>(pointer_v);
using custom_cast3 = double (*)[n2][n1];
custom_cast3 r = reinterpret_cast<custom_cast3>(pointer_r);	
#else
double (*u)[n2][n1] = (double (*)[n2][n1])pointer_u;
double (*v)[n2][n1] = (double (*)[n2][n1])pointer_v;
double (*r)[n2][n1] = (double (*)[n2][n1])pointer_r;		
#endif

int i3, i2, i1;
double u1[M], u2[M];

if(timeron){
#pragma omp master
timer_start(T_RESID);
}
#pragma omp for
for(i3 = 1; i3 < n3-1; i3++){
for(i2 = 1; i2 < n2-1; i2++){
for(i1 = 0; i1 < n1; i1++){
u1[i1] = u[i3][i2-1][i1] + u[i3][i2+1][i1]
+ u[i3-1][i2][i1] + u[i3+1][i2][i1];
u2[i1] = u[i3-1][i2-1][i1] + u[i3-1][i2+1][i1]
+ u[i3+1][i2-1][i1] + u[i3+1][i2+1][i1];
}
for(i1 = 1; i1 < n1-1; i1++){
r[i3][i2][i1] = v[i3][i2][i1]
- a[0] * u[i3][i2][i1]

- a[2] * ( u2[i1] + u1[i1-1] + u1[i1+1] )
- a[3] * ( u2[i1-1] + u2[i1+1] );
}
}
}
if(timeron){
#pragma omp master
timer_stop(T_RESID);
}


comm3(r,n1,n2,n3,k);

if(debug_vec[0] >= 1){
#pragma omp single
rep_nrm(r,n1,n2,n3,(char*)"   resid",k);
}

if(debug_vec[2] >= k){
#pragma omp single
showall(r,n1,n2,n3);
}
}


static void rprj3(void* pointer_r, int m1k, int m2k, int m3k, void* pointer_s, int m1j, int m2j, int m3j, int k){
#ifdef __clang__
using custom_cast = double (*)[m2k][m1k];
custom_cast r = reinterpret_cast<custom_cast>(pointer_r);
using custom_cast2 = double (*)[m2j][m1j];
custom_cast2 s = reinterpret_cast<custom_cast2>(pointer_s);
#else
double (*r)[m2k][m1k] = (double (*)[m2k][m1k])pointer_r;
double (*s)[m2j][m1j] = (double (*)[m2j][m1j])pointer_s;		
#endif	

int j3, j2, j1, i3, i2, i1, d1, d2, d3, j;

double x1[M], y1[M], x2, y2;

if(timeron){
#pragma omp master
timer_start(T_RPRJ3);
}
if(m1k == 3){
d1 = 2;
}else{
d1 = 1;
}
if(m2k == 3){
d2 = 2;
}else{
d2 = 1;
}
if(m3k == 3){
d3 = 2;
}else{
d3 = 1;
}

#pragma omp for
for(j3 = 1; j3 < m3j-1; j3++){
i3 = 2*j3-d3;		
for(j2 = 1; j2 < m2j-1; j2++){
i2 = 2*j2-d2;
for(j1 = 1; j1 < m1j; j1++){
i1 = 2*j1-d1;				
x1[i1] = r[i3+1][i2][i1] + r[i3+1][i2+2][i1]
+ r[i3][i2+1][i1] + r[i3+2][i2+1][i1];
y1[i1] = r[i3][i2][i1] + r[i3+2][i2][i1]
+ r[i3][i2+2][i1] + r[i3+2][i2+2][i1];
}
for(j1 = 1; j1 < m1j-1; j1++){
i1 = 2*j1-d1;				
y2 = r[i3][i2][i1+1] + r[i3+2][i2][i1+1]
+ r[i3][i2+2][i1+1] + r[i3+2][i2+2][i1+1];
x2 = r[i3+1][i2][i1+1] + r[i3+1][i2+2][i1+1]
+ r[i3][i2+1][i1+1] + r[i3+2][i2+1][i1+1];
s[j3][j2][j1] =
0.5 * r[i3+1][i2+1][i1+1]
+ 0.25 * ( r[i3+1][i2+1][i1] + r[i3+1][i2+1][i1+2] + x2)
+ 0.125 * ( x1[i1] + x1[i1+2] + y2)
+ 0.0625 * ( y1[i1] + y1[i1+2] );
}
}
}
if(timeron){
#pragma omp master
timer_stop(T_RPRJ3);
}

j=k-1;
comm3(s,m1j,m2j,m3j,j);

if(debug_vec[0] >= 1){
#pragma omp single
rep_nrm(s,m1j,m2j,m3j,(char*)"   rprj3",k-1);	
}

if(debug_vec[4] >= k){
#pragma omp single
showall(s,m1j,m2j,m3j);
}
}

static void setup(int* n1, int* n2, int* n3, int k){
int j;

int ax, mi[MAXLEVEL+1][3];
int ng[MAXLEVEL+1][3];

ng[lt][0] = nx[lt];
ng[lt][1] = ny[lt];
ng[lt][2] = nz[lt];
for(ax = 0; ax < 3; ax++){
for(k = lt-1; k >= 1; k--){
ng[k][ax] = ng[k+1][ax]/2;
}
}
for(k = lt; k >= 1; k--){
nx[k] = ng[k][0];
ny[k] = ng[k][1];
nz[k] = ng[k][2];
}

for(k = lt; k >= 1; k--){
for (ax = 0; ax < 3; ax++){
mi[k][ax] = 2 + ng[k][ax];
}

m1[k] = mi[k][0];
m2[k] = mi[k][1];
m3[k] = mi[k][2];
}

k = lt;
is1 = 2 + ng[k][0] - ng[lt][0];
ie1 = 1 + ng[k][0];
*n1 = 3 + ie1 - is1;
is2 = 2 + ng[k][1] - ng[lt][1];
ie2 = 1 + ng[k][1];
*n2 = 3 + ie2 - is2;
is3 = 2 + ng[k][2] - ng[lt][2];
ie3 = 1 + ng[k][2];
*n3 = 3 + ie3 - is3;

ir[lt] = 0;
for(j = lt-1; j >= 1; j--){
ir[j] = ir[j+1]+ONE*m1[j+1]*m2[j+1]*m3[j+1];
}

if(debug_vec[1] >= 1){
printf(" in setup, \n");
printf("   k  lt  nx  ny  nz  n1  n2  n3 is1 is2 is3 ie1 ie2 ie3\n");
printf("%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d\n", 
k,lt,ng[k][0],ng[k][1],ng[k][2],*n1,*n2,*n3,is1,is2,is3,ie1,ie2,ie3);
}
}

static void showall(void* pointer_z, int n1, int n2, int n3){
#ifdef __clang__
using custom_cast = double (*)[n2][n1];
custom_cast z = reinterpret_cast<custom_cast>(pointer_z);	
#else
double (*z)[n2][n1] = (double (*)[n2][n1])pointer_z;	
#endif

int i1,i2,i3;
int m1, m2, m3;

m1 = min(n1,18);
m2 = min(n2,14);
m3 = min(n3,18);

printf("\n");
for(i3 = 0; i3 < m3; i3++){
for(i2 = 0; i2 < m2; i2++){
for(i1 = 0; i1 < m1; i1++){			
printf("%6.3f", z[i3][i2][i1]);
}
printf("\n");
}
printf(" - - - - - - - \n");
}
printf("\n");
}

static void zero3(void* pointer_z, int n1, int n2, int n3){
#ifdef __clang__
using custom_cast = double (*)[n2][n1];
custom_cast z = reinterpret_cast<custom_cast>(pointer_z);
#else
double (*z)[n2][n1] = (double (*)[n2][n1])pointer_z;
#endif

int i1, i2, i3;

#pragma omp for  
for(i3 = 0;i3 < n3; i3++){
for(i2 = 0; i2 < n2; i2++){
for(i1 = 0; i1 < n1; i1++){
z[i3][i2][i1] = 0.0;
}
}
}
}


static void zran3(void* pointer_z, int n1, int n2, int n3, int nx, int ny, int k){
#ifdef __clang__
using custom_cast = double (*)[n2][n1];
custom_cast z = reinterpret_cast<custom_cast>(pointer_z);
#else
double (*z)[n2][n1] = (double (*)[n2][n1])pointer_z;
#endif

int i0, m0, m1;

int i1, i2, i3, d1, e1, e2, e3;
double xx, x0, x1, a1, a2, ai;

double ten[2][MM], best;
int i, j1[2][MM], j2[2][MM], j3[2][MM];
int jg[2][MM][4];

a1 = power(A, nx);
a2 = power(A, nx*ny);

#pragma omp parallel
zero3(z, n1, n2, n3);

i = is1-2+nx*(is2-2+ny*(is3-2));

ai = power(A, i);
d1 = ie1 - is1 + 1;
e1 = ie1 - is1 + 2;
e2 = ie2 - is2 + 2;
e3 = ie3 - is3 + 2;
x0 = X;
randlc(&x0, ai);
for(i3 = 1; i3 < e3; i3++){
x1 = x0;
for(i2 = 1; i2 < e2; i2++){
xx = x1;
vranlc(d1, &xx, A, &(z[i3][i2][1]));
randlc(&x1,a1);
}
randlc(&x0, a2);
}


for(i = 0; i < MM; i++){
ten[1][i] = 0.0;
j1[1][i] = 0;
j2[1][i] = 0;
j3[1][i] = 0;
ten[0][i] = 1.0;
j1[0][i] = 0;
j2[0][i] = 0;
j3[0][i] = 0;
}
for(i3 = 1; i3 < n3-1; i3++){
for(i2 = 1; i2 < n2-1; i2++){
for(i1 = 1; i1 < n1-1; i1++){
if(z[i3][i2][i1] > ten[1][0]){
ten[1][0] = z[i3][i2][i1];
j1[1][0] = i1;
j2[1][0] = i2;
j3[1][0] = i3;
bubble(ten, j1, j2, j3, MM, 1);
}
if(z[i3][i2][i1] < ten[0][0]){
ten[0][0] = z[i3][i2][i1];
j1[0][0] = i1;
j2[0][0] = i2;
j3[0][0] = i3;
bubble(ten, j1, j2, j3, MM, 0);
}
}
}
}


i1 = MM - 1;
i0 = MM - 1; 
for(i = MM - 1; i >= 0; i--){
best = 0.0;
if(best < ten[1][i1]){
jg[1][i][0] = 0;
jg[1][i][1] = is1 - 2 + j1[1][i1];
jg[1][i][2] = is2 - 2 + j2[1][i1];
jg[1][i][3] = is3 - 2 + j3[1][i1];
i1 = i1-1;
}else{
jg[1][i][0] = 0;
jg[1][i][1] = 0;
jg[1][i][2] = 0;
jg[1][i][3] = 0;
}
best = 1.0;
if(best > ten[0][i0]){
jg[0][i][0] = 0;
jg[0][i][1] = is1 - 2 + j1[0][i0];
jg[0][i][2] = is2 - 2 + j2[0][i0];
jg[0][i][3] = is3 - 2 + j3[0][i0];
i0 = i0-1;
}else{
jg[0][i][0] = 0;
jg[0][i][1] = 0;
jg[0][i][2] = 0;
jg[0][i][3] = 0;
}
}
m1 = 0;
m0 = 0;

for(i3 = 0; i3 < n3; i3++){
for(i2 = 0; i2 < n2; i2++){
for(i1 = 0; i1 < n1; i1++){
z[i3][i2][i1] = 0.0;
}
}
}
for (i = MM-1; i >= m0; i--){
z[jg[0][i][3]][jg[0][i][2]][jg[0][i][1]] = -1.0;
}
for(i = MM-1; i >= m1; i--){
z[jg[1][i][3]][jg[1][i][2]][jg[1][i][1]] = +1.0;
}
#pragma omp parallel 
comm3(z, n1, n2, n3, k);
}
