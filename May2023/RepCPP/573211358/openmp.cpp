#include <omp.h>
#include "physics.h"
#include "shared.h"

int n_omp_threads;

void update(float *data, float *new_data) {
#pragma omp parallel for
for (int i = 1; i < (size - 1); i++){
for (int j = 1; j < (size - 1); j++){
int idx = i * size + j;
if (fire_area[idx]) new_data[idx] = fire_temp;
else {
float up = data[idx - size];
float down = data[idx + size];
float left = data[idx - 1];
float right = data[idx + 1];
float new_val = (up + down + left + right) / 4;
new_data[idx] = new_val;
}
}
}
}

void maintain_wall(float *data) {
#pragma omp parallel for
for (int i = 0; i < size; i++){
data[i] = wall_temp;
data[i * size] = wall_temp;
data[i * size + size - 1] = wall_temp;
data[(size - 1) * size + i] = wall_temp;
}
}


void master() {
data = new float[size * size];
new_data = new float[size * size];
fire_area = new bool[size * size];

generate_fire_area(fire_area);
initialize(data);

while (count <= max_iteration){
t1 = std::chrono::high_resolution_clock::now();

omp_set_num_threads(n_omp_threads);
update(data, new_data);
maintain_wall(new_data);
swap(data, new_data);

t2 = std::chrono::high_resolution_clock::now();
double this_time = std::chrono::duration<double>(t2 - t1).count();
if (DEBUG) printf("Iteration %d, elapsed time: %.6f\n", count, this_time);
total_time += this_time;

#ifdef GUI
data2pixels(data, pixels);
plot(pixels);
#endif

count++;
}

printf("Converge after %d iterations, elapsed time: %.6f, average computation time: %.6f\n", count-1, total_time, (double) total_time / (count-1));

delete[] data;
delete[] new_data;
delete[] fire_area;
}


int main(int argc, char *argv[]){

size = atoi(argv[1]);
n_omp_threads = atoi(argv[2]);

#ifdef GUI
glutInit(&argc, argv);
glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
glutInitWindowPosition(0, 0);
glutInitWindowSize(window_size, window_size);
glutCreateWindow("Heat Distribution Simulation OpenMP Implementation");
gluOrtho2D(0, resolution, 0, resolution);
#endif

master();

printf("Student ID: 119020059\n");
printf("Name: Xinyu Xie\n");
printf("Assignment 4: Heat Distribution OpenMP Implementation\n");
printf("Problem Size: %d\n", size);
printf("Number of Cores: %d\n", n_omp_threads);

return 0;

}


