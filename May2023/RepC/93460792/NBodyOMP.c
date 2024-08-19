



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define n 800   
#define E 1.0
#define Rm 0.001
#define L 1.0
#define T 10
#define dt 0.001
#define FILE_NAME "output.txt"



typedef struct{
double pos[2];
double vel[2];
double acc[2];
double npos[2];
double mass;
}particle;

static particle parts[n];

typedef struct{
double pos[2];
double vel[2];
} phist;

void copyp(particle p1,phist *p2){
p2->pos[0]=p1.pos[0];
p2->pos[1]=p1.pos[1];
p2->vel[0]=p1.vel[0];
p2->vel[1]=p1.vel[1];
}

void showstat(){
for(int i=0;i<n;i++){
printf("%f - %f \n",parts[i].pos[0],parts[i].pos[1]);
}
}

double unirand(double min, double max) 
{
double range = (max - min); 
double div = RAND_MAX / range;
return min + (rand() / div);
}

double* force(int i,int j){
double dx=parts[j].npos[0]-parts[i].npos[0];
double dy=parts[j].npos[1]-parts[i].npos[1];
double dist=hypot(dx,dy);
double aux=Rm/dist;
aux=pow(aux,6.0);
aux=(aux*aux-aux)*12.0*E/(dist*dist);
static double f[2];
f[0]=aux*dx;
f[1]=aux*dy;
return f;
}


void vecadd(double* v1,double* v2){
*v1+=*v2;
*(v1+1)+=*(v2+1);
}

void vecadds(double* v1,double* v2,double a){
*v1+=(*v2)*a;
*(v1+1)+=(*(v2+1))*a;
}

void vecset(double* v1,double* v2){
*v1=*v2;
*(v1+1)=*(v2+1);
}

void vecsets(double* v1,double* v2,double a){
*v1=(*v2)*a;
*(v1+1)=(*(v2+1))*a;
}

int main(int argc, char** argv) {


static phist data[n*T];
for(int i=0;i<n;i++){
particle *p=&parts[i];
(*p).pos[0]=unirand(0.0,L);
(*p).pos[1]=unirand(0.0,L);
(*p).npos[0]=unirand(0.0,L);
(*p).npos[1]=unirand(0.0,L);
(*p).mass=1.0;
}
for(int i=0;i<T;i++){
omp_set_num_threads(8);
#pragma omp parallel
{
int id=omp_get_thread_num();
for(int jz=0;jz<(n/8);jz++){
int j=(n/8)*id+jz;
vecadds(parts[j].vel,parts[j].acc,0.5*dt);
vecadds(parts[j].npos,parts[j].vel,dt);
double tf[2]={0.0,0.0};
for(int k=0;k<n;k++){
if(k==j) continue;
vecadd(tf,force(j,k));
}
vecsets(parts[j].acc,tf,(1/parts[j].mass));
vecadds(parts[j].vel,parts[j].acc,0.5*dt);
}
}


for(int j=0;j<n;j++){
vecset(parts[j].pos,parts[j].npos);
copyp(parts[j],&data[i*T+j]);
}

}


FILE *fp;
fp = fopen("/tmp/data.txt", "w+");
fprintf(fp,"%d %d %.15f %.15f %.15f %.15f\n",n,T,dt,L,E,Rm);
for(int i=0;i<T;i++){
for(int j=0;j<n;j++){
fprintf(fp,"%.15f %.15f %.15f %.15f\n",data[T*i+j].pos[0],
data[T*i+j].pos[1],data[T*i+j].vel[0],data[T*i+j].vel[1]);
}
}
fclose(fp);

return (EXIT_SUCCESS);
}

