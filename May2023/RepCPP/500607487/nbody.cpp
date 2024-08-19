#include <iostream>
using namespace std;
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <omp.h>

struct cartesian3D {
double x;
double y;
double z;

};

void print_cartesian3D(cartesian3D point) {
cout << "(" << point.x << ", " << point.y << ", " << point.z << ")" << endl;
}

struct Body {
cartesian3D r; 
cartesian3D v; 
double m; 
};

void print_body(Body b) {
cout << "r: (" << b.r.x << " ," << b.r.y << " ," << b.r.z << ")" << "  v:" << b.v.x << " ," << b.v.y << " ," << b.v.z << ") m: " << b.m << endl;
}

void write_to_file(Body* bodies, string filename, int n, int t) { 
ofstream file;
file.open(filename);

for (int i = 0; i < n; i++) {
if (i == 0) {
file << "[[" << bodies[i].r.x << "," << bodies[i].r.y << "," << bodies[i].r.z << "],";
}
else if (i == n - 1) {
file << "[" << bodies[i].r.x << "," << bodies[i].r.y << "," << bodies[i].r.z << "]]";
}
else {
file << "[" << bodies[i].r.x << "," << bodies[i].r.y << "," << bodies[i].r.z << "],";
}
}

file.close();
}

void initialize_bodies(Body* bodies, int n, string initialization_type) {
int i;
double phi, theta;

#pragma omp parallel for default(none) private(i, phi, theta) shared(bodies, n, initialization_type) schedule(guided)
for (i = 0; i < n; i++) {
phi = 2 * M_PI * (double)rand() / (double)RAND_MAX;
theta = acos((double)rand() / (double)RAND_MAX);

if (initialization_type == "random") {
bodies[i].r.x = (double)rand() / (double)RAND_MAX;
bodies[i].r.y = (double)rand() / (double)RAND_MAX;
bodies[i].r.z = (double)rand() / (double)RAND_MAX;

bodies[i].v.x = (double)rand() / (double)RAND_MAX;
bodies[i].v.y = (double)rand() / (double)RAND_MAX;
bodies[i].v.z = (double)rand() / (double)RAND_MAX;

bodies[i].m = (double)rand() / (double)RAND_MAX;
}
else if (initialization_type == "uniform") {
bodies[i].r.x = (double)(i + 1) / (double)n;
bodies[i].r.y = (double)(i + 1) / (double)n;
bodies[i].r.z = (double)(i + 1) / (double)n;

bodies[i].v.x = (double)(i + 1) / (double)n;
bodies[i].v.y = (double)(i + 1) / (double)n;
bodies[i].v.z = (double)(i + 1) / (double)n;

bodies[i].m = (double)(i + 1) / (double)n;
}
else if (initialization_type == "elipsoid") {
bodies[i].r.x = cos(theta) * sin(phi);
bodies[i].r.y = sin(theta) * sin(phi);
bodies[i].r.z = cos(phi);

bodies[i].v.x = -sin(theta) * sin(phi);
bodies[i].v.y = cos(theta) * sin(phi);
bodies[i].v.z = -sin(phi);

bodies[i].m = 1.0 / (1.0 + bodies[i].r.x * bodies[i].r.x + bodies[i].r.y * bodies[i].r.y + bodies[i].r.z * bodies[i].r.z);
}
else if (initialization_type == "galaxy") {
bodies[i].r.x = cos(phi);
bodies[i].r.y = (i + 1)*sin(phi);
bodies[i].r.z = 1;

bodies[i].v.x = -sin(phi);
bodies[i].v.y = cos(phi);
bodies[i].v.z = 1.0;

bodies[i].m = 1.0 / (1.0 + bodies[i].r.x * bodies[i].r.x + bodies[i].r.y * bodies[i].r.y + bodies[i].r.z * bodies[i].r.z);
}
else {
assert("Invalid initialization type");
exit(1);
}        
}    
}

void nbody(int n, double dt, int N, double G, string initialization_type){
Body* bodies = new Body[n];
initialize_bodies(bodies, n, initialization_type);
cartesian3D F_ij;

for (int t = 0; t < N; t++) {
if (t % 10 == 0) {
char filename[100];
int dummy_var = sprintf(filename, "./output/t_%d.txt", t);
write_to_file(bodies, filename, n, t);
}        

double t1;
double t2;

t1 = omp_get_wtime();

int i = 0;
int j = 0;
double r_mag;
cartesian3D F_i;
#pragma omp parallel for default(none) private(i, j, F_ij, F_i, r_mag) shared(bodies, G, dt, n) schedule(guided) 
for (i = 0; i < n; i++) {
F_i = {0, 0, 0};

for (j = 0; j < n; j++) {
r_mag = sqrt(pow(bodies[j].r.x - bodies[i].r.x, 2) + pow(bodies[j].r.y - bodies[i].r.y, 2) + pow(bodies[j].r.z - bodies[i].r.z, 2));

if (r_mag > 0.0) {
F_ij.x = G * bodies[i].m * bodies[j].m * (bodies[j].r.x - bodies[i].r.x) / pow(r_mag, 3);
F_ij.y = G * bodies[i].m * bodies[j].m * (bodies[j].r.y - bodies[i].r.y) / pow(r_mag, 3);
F_ij.z = G * bodies[i].m * bodies[j].m * (bodies[j].r.z - bodies[i].r.z) / pow(r_mag, 3);

F_i.x += F_ij.x;
F_i.y += F_ij.y;
F_i.z += F_ij.z;
}
}
bodies[i].v.x += dt * F_i.x / bodies[i].m;
bodies[i].v.y += dt * F_i.y / bodies[i].m;
bodies[i].v.z += dt * F_i.z / bodies[i].m;
}
#pragma omp parallel for default(none) private(i) shared(bodies, dt, n) schedule(guided) 
for (i = 0; i < n; i++) {
bodies[i].r.x += dt * bodies[i].v.x;
bodies[i].r.y += dt * bodies[i].v.y;
bodies[i].r.z += dt * bodies[i].v.z;
}

t2 = omp_get_wtime();

int grind_rate = floor(1.0 / (t2 - t1));
printf("Timestep: %d - Grind Rate: %d iter/secs\n", t, grind_rate);
}
}

int main(int argc, char** argv) {
int n = stoi(argv[1]); 
double dt = stod(argv[2]); 
int N = stoi(argv[3]); 
double G = stod(argv[4]); 
int num_threads = stoi(argv[5]); 
string initialization_type = argv[6]; 

cout << "Simulation Parameters:" << endl;
cout << "Number of particles: " << n << endl;
cout << "Timestep: " << dt << endl;
cout << "Number of timesteps: " << N << endl;
cout << "Gravitational constant: " << G << endl;
cout << "Initialization type: " << initialization_type << endl;

omp_set_num_threads(num_threads);
cout << "Number of threads = " << num_threads << "\n" << endl;

double ts;
double tf;

ts = omp_get_wtime();
nbody(n, dt, N, G, initialization_type);
tf = omp_get_wtime();

cout << "Time taken = " << tf-ts << " seconds" << "\n" << endl;

return 0;

}