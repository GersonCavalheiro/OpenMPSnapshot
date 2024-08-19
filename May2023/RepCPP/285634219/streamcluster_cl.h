

#define THREADS_PER_BLOCK 256
#define MAXBLOCKS 65536

typedef struct {
float weight;
long assign;  
float cost;  
} Point_Struct;


float *work_mem_h;
float *coord_h;
float *gl_lower;
Point_Struct *p_h;

static int c;      

float pgain( long x, Points *points, float z, long int *numcenters, 
int kmax, bool *is_center, int *center_table, char *switch_membership,
double *serial, double *cpu_gpu_memcpy, double *memcpy_back,
double *gpu_malloc, double *kernel_time) {

float gl_cost = 0;
try{
#ifdef PROFILE_TMP
double t1 = gettime();
#endif
int K = *numcenters;   
int num = points->num; 
int dim = points->dim; 
kmax++;

int count = 0;
for( int i=0; i<num; i++){
if( is_center[i] )
center_table[i] = count++;
}

#ifdef PROFILE_TMP
double t2 = gettime();
*serial += t2 - t1;
#endif


if( c == 0 ) {
#ifdef PROFILE_TMP
double t3 = gettime();
#endif
coord_h = (float*) malloc( num * dim * sizeof(float));                
gl_lower = (float*) malloc( kmax * sizeof(float) );
work_mem_h = (float*) malloc ((kmax+1)*num*sizeof(float));  
p_h = (Point_Struct*)malloc(num*sizeof(Point_Struct));  

for(int i=0; i<dim; i++){
for(int j=0; j<num; j++)
coord_h[ (num*i)+j ] = points->p[j].coord[i];
}
#ifdef PROFILE_TMP    
double t4 = gettime();
*serial += t4 - t3;
#endif

#pragma omp target enter data map(alloc: coord_h[0:dim*num],\
center_table[0:num],\
work_mem_h[0:(kmax+1)*num], \
switch_membership[0:num], \
p_h[0:num])

#ifdef PROFILE_TMP
double t5 = gettime();
*gpu_malloc += t5 - t4;
#endif

#pragma omp target update to(coord_h[0:num*dim])

#ifdef PROFILE_TMP
double t6 = gettime();
*cpu_gpu_memcpy += t6 - t4;
#endif
}    

#ifdef PROFILE_TMP
double t100 = gettime();
#endif

for(int i=0; i<num; i++){
p_h[i].weight = ((points->p)[i]).weight;
p_h[i].assign = ((points->p)[i]).assign;
p_h[i].cost = ((points->p)[i]).cost;  
}

#ifdef PROFILE_TMP
double t101 = gettime();
*serial += t101 - t100;
#endif
#ifdef PROFILE_TMP
double t7 = gettime();
#endif

#pragma omp target update to(p_h[0:num])
#pragma omp target update to(center_table[0:num])

#ifdef PROFILE_TMP
double t8 = gettime();
*cpu_gpu_memcpy += t8 - t7;
#endif



const size_t smSize = 256; 

#ifdef PROFILE_TMP
double t9 = gettime();
#endif

#pragma omp target teams distribute parallel for thread_limit(256)
for (int i = 0; i < num; i++) switch_membership[i] = 0;

#pragma omp target teams distribute parallel for thread_limit(256)
for (int i = 0; i < num*(K+1); i++) work_mem_h[i] = 0;

int work_group_size = THREADS_PER_BLOCK;
int work_items = num;
if(work_items%work_group_size != 0)  
work_items = work_items + (work_group_size-(work_items%work_group_size));

#pragma omp target teams num_teams(work_items/work_group_size) thread_limit(work_group_size)
{
float coord_s_acc[smSize];
#pragma omp parallel
{
#include "kernel.h"
}
}

#ifdef PROFILE_TMP
double t10 = gettime();
*kernel_time += t10 - t9;
#endif


#pragma omp target update from(switch_membership[0:num])
#pragma omp target update from(work_mem_h[0:num*(K+1)])

#ifdef PROFILE_TMP
double t11 = gettime();
*memcpy_back += t11 - t10;
#endif


int numclose = 0;
gl_cost = z;


for(int i=0; i < num; i++){
if( is_center[i] ) {
float low = z;
for( int j = 0; j < num; j++ )
low += work_mem_h[ j*(K+1) + center_table[i] ];
gl_lower[center_table[i]] = low;

if ( low > 0 ) {
numclose++;        
work_mem_h[i*(K+1)+K] -= low;
}
}
gl_cost += work_mem_h[i*(K+1)+K];
}


if ( gl_cost < 0 ) {
for(int i=0; i<num; i++){

bool close_center = gl_lower[center_table[points->p[i].assign]] > 0 ;
if ( (switch_membership[i]=='1') || close_center ) {
points->p[i].cost = points->p[i].weight * dist(points->p[i], points->p[x], points->dim);
points->p[i].assign = x;
}
}

for(int i=0; i<num; i++){
if( is_center[i] && gl_lower[center_table[i]] > 0 )
is_center[i] = false;
}

is_center[x] = true;
*numcenters = *numcenters +1 - numclose;
}
else
gl_cost = 0;

#ifdef PROFILE_TMP
double t12 = gettime();
*serial += t12 - t11;
#endif
c++;
}
catch(string msg){
printf("--cambine:%s\n", msg.c_str());
exit(-1);    
}
catch(...){
printf("--cambine: unknow reasons in pgain\n");
}

#ifdef DEBUG
FILE *fp = fopen("data_debug.txt", "a");
fprintf(fp,"%d, %f\n", c, gl_cost);
fclose(fp);
#endif
return -gl_cost;
}
