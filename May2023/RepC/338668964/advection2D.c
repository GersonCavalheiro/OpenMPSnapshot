#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
float calc_vel_x(float height);
int main()
{
const int NX = 1000;	   
const int NY = 1000;	   
const float MIN_X = 0.0f;  
const float MAX_X = 30.0f; 
const float MIN_Y = 0.0f;  
const float MAX_Y = 30.0f; 
const float CENTER_X = 3.0f;			  
const float CENTER_Y = 15.0f;			  
const float SIGMA_X  = 1.0f;			  
const float SIGMA_Y  = 5.0f;			  
const float SIGMA_X2 = SIGMA_X * SIGMA_X; 
const float SIGMA_Y2 = SIGMA_Y * SIGMA_Y; 
const float BVAL_LEFT  = 0.0f;	
const float BVAL_RIGHT = 0.0f; 	
const float BVAL_LOWER = 0.0f; 	
const float BVAL_UPPER = 0.0f; 	
const float CFL = 0.9f;	  
const int N_STEPS = 800;  
const float VEL_X = calc_vel_x(MAX_Y); 
const float VEL_Y = 0.0f;			   
const float DIST_X = (MAX_X - MIN_X) / ((float)NX);
const float DIST_Y = (MAX_Y - MIN_Y) / ((float)NY);
const float TIME_STEP = CFL / ((fabs(VEL_X) / DIST_X) + (fabs(VEL_Y) / DIST_Y));
float x_axis[NX + 2];			
float y_axis[NX + 2];			
float u_val[NX + 2][NY + 2];	
float  roc_u[NX + 2][NY + 2];	
float *vavg_u = (float *)calloc(NX, sizeof(float)); 
float sq_x2; 
float sq_y2; 
printf("Grid spacing dx     = %g\n", DIST_X);
printf("Grid spacing dy     = %g\n", DIST_Y);
printf("CFL number          = %g\n", CFL);
printf("Time step           = %g\n", TIME_STEP);
printf("No. of time steps   = %d\n", N_STEPS);
printf("End time            = %g\n", TIME_STEP * (float)N_STEPS);
printf("Distance advected x = %g\n", VEL_X * TIME_STEP * (float)N_STEPS);
printf("Distance advected y = %g\n", VEL_Y * TIME_STEP * (float)N_STEPS);
#pragma omp parallel default(none) shared(x_axis, y_axis, u_val) firstprivate(NX, NY, DIST_X, DIST_Y, CENTER_X, CENTER_Y, SIGMA_X2, SIGMA_Y2)
{
#pragma omp for
for (int i = 0; i < NX + 2; i++)
{
x_axis[i] = ((float)i - 0.5f) * DIST_X;
}
#pragma omp for
for (int j = 0; j < NY + 2; j++)
{
y_axis[j] = ((float)j - 0.5f) * DIST_Y;
}
#pragma omp for private(sq_x2, sq_y2)
for (int i = 0; i < NX + 2; i++)
{
for (int j = 0; j < NY + 2; j++)
{
sq_x2 = (x_axis[i] - CENTER_X) * (x_axis[i] - CENTER_X);
sq_y2 = (y_axis[j] - CENTER_Y) * (y_axis[j] - CENTER_Y);
u_val[i][j] = exp(-1.0f * ((sq_x2 / (2.0f * SIGMA_X2)) + (sq_y2 / (2.0f * SIGMA_Y2))));
}
}
}
FILE *initial_file;
initial_file = fopen("initial.dat", "w");
for (int i = 0; i < NX + 2; i++)
{
for (int j = 0; j < NY + 2; j++)
{
fprintf(initial_file, "%g %g %g\n", x_axis[i], y_axis[j], u_val[i][j]);
}
}
fclose(initial_file);
#pragma omp parallel default(none) shared(x_axis, y_axis, u_val, roc_u, vavg_u) firstprivate(NX, NY, DIST_X, DIST_Y, VEL_Y, N_STEPS, TIME_STEP, BVAL_LEFT, BVAL_RIGHT, BVAL_LOWER, BVAL_UPPER)
{
for (int m = 0; m < N_STEPS; m++)
{
#pragma omp for
for (int j = 0; j < NY + 2; j++)
{
u_val[0][j] = BVAL_LEFT;
u_val[NX + 1][j] = BVAL_RIGHT;
}
#pragma omp for
for (int i = 0; i < NX + 2; i++)
{
u_val[i][0] = BVAL_LOWER;
u_val[i][NY + 1] = BVAL_UPPER;
}
#pragma omp for
for (int i = 1; i < NX + 1; i++)
{
for (int j = 1; j < NY + 1; j++)
{
roc_u[i][j] = -calc_vel_x(y_axis[j]) * (u_val[i][j] - u_val[i - 1][j]) / DIST_X 
- VEL_Y * (u_val[i][j] - u_val[i][j - 1]) / DIST_Y;
}
}
#pragma omp for
for (int i = 1; i < NX + 1; i++)
{
for (int j = 1; j < NY + 1; j++)
{
u_val[i][j] = u_val[i][j] + roc_u[i][j] * TIME_STEP;
vavg_u[i - 1] += u_val[i][j];	
}
vavg_u[i - 1] /= NY; 
}
}
}
FILE *final_file;
final_file = fopen("final.dat", "w");
for (int i = 0; i < NX + 2; i++)
{
for (int j = 0; j < NY + 2; j++)
{
fprintf(final_file, "%g %g %g\n", x_axis[i], y_axis[j], u_val[i][j]);
}
}
fclose(final_file);
FILE *v_averaged_file;
v_averaged_file = fopen("v_averaged.dat", "w");
for (int i = 0; i < NX; i++)
{
fprintf(v_averaged_file, "%g %g\n", x_axis[i], vavg_u[i]);
}
fclose(v_averaged_file);
free(vavg_u);
return 0;
}
float calc_vel_x(float height)
{
const float F_VEL    = 0.2f;  
const float R_LEN    = 1.0f;  
const float VK_CONST = 0.41f; 
float vel_x = 0;
if (height > R_LEN)
{
vel_x = (F_VEL / VK_CONST) * logf(height / R_LEN);
}
return vel_x;
}