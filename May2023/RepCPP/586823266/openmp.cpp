#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "./headers/physics.h"
#include "./headers/logger.h"


int n_body;
int n_iteration;

int n_omp_threads;
std::chrono::high_resolution_clock::time_point start;
std::chrono::high_resolution_clock::time_point end;

void generate_data(double *m, double *x,double *y,double *vx,double *vy, int n) {
srand((unsigned)time(NULL));
for (int i = 0; i < n; i++) {
m[i] = rand() % max_mass + 1.0f;
x[i] = 1500.0f + rand() % (bound_x / 4);
y[i] = 1500.0f + rand() % (bound_y / 4);
vx[i] = 0.0f;
vy[i] = 0.0f;
}
}

void update_position(double *x, double *y, double *vx, double *vy, int i) {
x[i] = x[i] + vx[i] * dt;
y[i] = y[i] + vy[i] * dt;

if (x[i] <= sqrt(radius2) || x[i] >= bound_x - sqrt(radius2)) vx[i] = -vx[i];
if (y[i] <= sqrt(radius2) || y[i] >= bound_y - sqrt(radius2)) vy[i] = -vy[i];


}

void update_velocity(double *m, double *x, double *y, double *vx, double *vy, int i) {
double deltaX, deltaY, distance, acceleration;
for (int j = 0; j < n_body; j++){

if (i == j) continue;

deltaX = x[j] - x[i];
deltaY = y[j] - y[i];
distance = sqrt((deltaX * deltaX) + (deltaY * deltaY));

if (distance < 2 * sqrt(radius2)) {
vx[i] = -vx[i];
vy[i] = -vy[i];
continue;
}

acceleration = gravity_const * m[j] / (distance * distance);
vx[i] += acceleration * deltaX / distance * dt;
vy[i] += acceleration * deltaY / distance * dt;
}

}


void master() {
double* m = new double[n_body];
double* x = new double[n_body];
double* y = new double[n_body];
double* vx = new double[n_body];
double* vy = new double[n_body];

generate_data(m, x, y, vx, vy, n_body);


for (int i = 0; i < n_iteration; i++){

omp_set_num_threads(n_omp_threads);
#pragma omp parallel for
for (int i = 0; i < n_body; i++) {
update_velocity(m, x, y, vx, vy, i);
}

omp_set_num_threads(n_omp_threads);
#pragma omp parallel for
for (int i = 0; i < n_body; i++) {
update_position(x, y, vx, vy, i);
}




#ifdef GUI
glClear(GL_COLOR_BUFFER_BIT);
glColor3f(1.0f, 0.0f, 0.0f);
glPointSize(2.0f);
glBegin(GL_POINTS);
double xi;
double yi;
for (int i = 0; i < n_body; i++){
xi = x[i];
yi = y[i];
glVertex2f(xi, yi);
}
glEnd();
glFlush();
glutSwapBuffers();
#else

#endif
}

delete[] m;
delete[] x;
delete[] y;
delete[] vx;
delete[] vy;

}


int main(int argc, char *argv[]){

n_body = atoi(argv[1]);
n_iteration = atoi(argv[2]);
n_omp_threads = atoi(argv[3]);

#ifdef GUI
glutInit(&argc, argv);
glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
glutInitWindowPosition(0, 0);
glutInitWindowSize(500, 500);
glutCreateWindow("N Body Simulation Sequential Implementation");
glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
gluOrtho2D(0, bound_x, 0, bound_y);
#endif
start = std::chrono::high_resolution_clock::now();
master();
end = std::chrono::high_resolution_clock::now();

printf("Student ID: 119010221\n"); 
printf("Name: Haoyan Luo\n"); 
printf("Assignment 3: N Body Simulation OpenMP Implementation\n");
std::chrono::duration<double> total_time = end - start;
printf("Total time: %.3f\n", total_time);

return 0;

}


