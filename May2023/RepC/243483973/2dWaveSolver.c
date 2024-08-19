#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>




double I(double x, double y){

return x*(10-x)*y*(10-y);
}

double V(double x, double y){

return 0.5*x*(10-x)*y*(10-y);
}

double f(double x, double y, double t){

return 2*(1 + 0.5*t)*(y*(10-y)+x*(10-x));
}

double analytical(double x, double y, double t){

return x*(10-x)*y*(10-y)*(1 + 0.5*t);
}





int main(){

int i,j,n,w;

double Lx = 10,Ly = 10;

int	Nx = 40,Ny = 40;

double	T = 20;
int Nt = 250;  

double r[Nx][Ny];

FILE *f0,*f5,*f10,*f20,*fspeed,*feff; 



double u[Nx][Ny],u_1[Nx][Ny],u_2[Nx][Ny],
x[Nx],y[Ny],t[Nt],
dx,dy,dt,
g,
TimeStart,TimeEnd,
elapsed[4],
central[4],
efficiency[4],
speedup[4];

int thread_num[] = {1,1,2,4,8}; 




dx = Lx/(Nx-1);
dy = Ly/(Ny-1);
dt = T/(Nt-1);




for (w=0;w<5;w++) 


{


omp_set_num_threads(thread_num[w]); 


TimeStart = omp_get_wtime();

#pragma omp parallel private(i,j,n) shared(x,dx,y,dy,t,dt,u,u_1,u_2,r,Nx,Ny,Nt,g)
{

#pragma omp for
for(i=0;i<Nx;i++){
x[i] = 0 + i*dx;
}
#pragma omp for
for(j=0;j<Ny;j++){
y[j] = 0 + j*dy;
}
#pragma omp for 
for(n=0;n<Nt;n++){
t[n] = 0 +n*dt;
}

g = pow(dt/dx,2);

if (w==0) 
{
f0 = fopen("data1.txt","w");
}

#pragma omp for collapse(2)
for(i=0;i<Nx;i++) {
for(j=0;j<Ny;j++){

u_1[i][j] = I(x[i],y[j]);
r[i][j] = analytical(x[i],y[j],0);


if (w==0) 
{
fprintf(f0, "%f %f %f %f \n\n", x[i],y[j],u_1[i][j],r[i][j]); 
}

}


}



#pragma omp for collapse(2)
for(i=1;i<Nx;i++) {
for(j=1;j<Ny;j++){

u[i][j] = u_1[i][j]*(1-2*g)+dt*V(x[i],y[j])+0.5*g*(u_1[i+1][j]+u_1[i][j+1]
+ u_1[i-1][j]+u_1[i][j-1]) + 0.5*pow(dt,2)*f(x[i],y[j],0);
}

}

#pragma omp for
for(i=0;i<Nx;i++) {
u[i][0] = 0;
u[i][39] = 0;}


#pragma omp for
for(j=0;j<Nx;j++) {
u[0][j] = 0;
u[39][j] = 0;}



#pragma omp for collapse(2)
for(i=0;i<Nx;i++) {
for(j=0;j<Ny;j++){

u_2[i][j]=u_1[i][j];
u_1[i][j]=u[i][j];

}
}


for(n=1;n<Nt;n++){
#pragma omp for collapse(2)
for(i=1;i<Nx;i++) {
for(j=1;j<Ny;j++){

u[i][j]= pow(dt,2)*f(x[i],y[j],t[n]) + g*(u_1[i+1][j]+u_1[i][j+1]
+u_1[i-1][j]+u_1[i][j-1]) + 2*u_1[i][j]*(1-2*g) -u_2[i][j];

r[i][j] = analytical(x[i],y[j],t[n]);

}
}



#pragma omp for
for(i=0;i<Nx;i++) {
u[i][0] = 0;
u[i][39] = 0;}

#pragma omp for
for(j=0;j<Nx;j++) {
u[0][j] = 0;
u[39][j] = 0;}



if (w==0){

if (n==60){


f5 = fopen("data5.txt","w");


#pragma omp for collapse(2)
for(i=0;i<Nx;i++) {
for(j=0;j<Ny;j++){
fprintf(f5, "%f %f %f %f \n\n ",x[i],y[j],u[i][j],r[i][j]);
}
}

}
else if (n==120){


f10 = fopen("data10.txt","w");

#pragma omp for collapse(2)
for(i=0;i<Nx;i++) {
for(j=0;j<Ny;j++){
fprintf(f10, "%f %f %f %f \n\n",x[i],y[j],u[i][j],r[i][j]);
}
}		

}
else if (n==Nt-1){


f20 = fopen("data20.txt","w");

#pragma omp for collapse(2) 
for(i=0;i<Nx;i++) {
for(j=0;j<Ny;j++){
fprintf(f20, "%f %f %f %f \n\n",x[i],y[j],u[i][j],r[i][j]);
}
}

}
}

#pragma omp for collapse(2)
for(i=0;i<Nx;i++) {
for(j=0;j<Ny;j++){
u_2[i][j]=u_1[i][j];
u_1[i][j]=u[i][j];
}

}
}

}



TimeEnd = omp_get_wtime();




if (w>=1)
{	
central[w-1] = u[Nx/2][Ny/2];
elapsed[w-1] = TimeEnd-TimeStart;
if (w>1){
speedup[w-1] = elapsed[0]/elapsed[w-1]; 
efficiency[w-1] = speedup[w-1]/thread_num[w];
}
else {
speedup[w-1] = 1.;
efficiency[w-1] = 1.;
}


printf("Total threads = %d\n\n Time passed = %f\n Parallel Speedup = %f\n Parallel Efficiency = %f\n \
Central Value = %f\n\n\n\n",thread_num[w],elapsed[w-1],speedup[w-1],efficiency[w-1],central[w-1]);


}



}


fspeed = fopen("speed.txt","w");
feff = fopen("efficiency.txt","w");
for(w=0;w<4;w++)
{
fprintf(fspeed,"%f %d\n",speedup[w],thread_num[w+1]);
fprintf(feff,"%f %d\n",efficiency[w],thread_num[w+1]);

}


return 0; 

}
