#include <stdio.h>
#include<omp.h>
#include <stdlib.h>

int main()
{
double x_len = 2.0;         
int x_points = 101;             
double del_x = x_len/(x_points-1);      

int num_itr = 1500;                              
double sigma = 0.2;
double nu = 0.3;
double del_t = (sigma/nu)*(del_x*del_x);        

double x[x_points];
double u[x_points];
double u_new[x_points];
#pragma omp parallel for
for (int i = 0; i < x_points; i++){
x[i] = i * del_x;                                   

if (x[i] > 0.5 && x[i] < 1.0){                  
u[i] = 2.0;
u_new[i] = 2.0;
}
else{
u[i] = 1.0;
u_new[i] = 1.0;
}
}

printf("\n x co-ordinates at the grid points are: \n \t");
for (int i = 0; i < x_points; i++){
printf("%f \t", x[i]);
}

printf("\n Initial u velocity at the grid points are: \n \t");
for (int i = 0; i < x_points; i++){
printf("%f \t", u[i]);
}

double parr_start_time = omp_get_wtime();
#pragma omp parallel
for (int it = 0; it < num_itr; it++){
double temp_next;
double temp_prev;

#pragma omp for nowait private(temp_next, temp_prev)
for (int i = 1; i < x_points-1; i++){
temp_next = u[i+1];
temp_prev = u[i-1];
u_new[i] = u[i] + (nu  * del_t/(del_x * del_x))*(temp_next + temp_prev - 2*u[i]);
}

#pragma omp for
for (int i = 0; i < x_points; i++){
u[i] = u_new[i];
}
}
double parr_end_time = omp_get_wtime();

printf("\n Final u velocity at the grid points are: \n \t");
for (int i = 0; i < x_points; i++){
printf("%f \t", u[i]);
}
printf("\n Parallel computing time taken: %f \n \t", parr_end_time - parr_start_time);



for (int i = 0; i < x_points; i++){
x[i] = i * del_x;                                   

if (x[i] > 0.5 && x[i] < 1.0){                  
u[i] = 2.0;
u_new[i] = 2.0;
}
else{
u[i] = 1.0;
u_new[i] = 1.0;
}
}

double ser_start_time = omp_get_wtime();
for (int it = 0; it < num_itr; it++){

for (int i = 1; i < x_points-1; i++){
u_new[i] = u[i] + (nu  * del_t/(del_x * del_x))*(u[i+1] + u[i-1] - 2*u[i]);
}

for (int i = 0; i < x_points; i++){
u[i] = u_new[i];
}
}
double ser_end_time = omp_get_wtime();

printf("\n Serial computing time taken: %f \n", ser_end_time - ser_start_time);

printf("\n The speed-up is : %f \n", (ser_end_time - ser_start_time)/(parr_end_time - parr_start_time));

return 0;
}
