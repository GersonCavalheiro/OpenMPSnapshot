#include <iostream>
using namespace std;
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include<omp.h>
#include <string>

void write_to_file(double** const& C_n, int const& N, string const& filename) {
ofstream out(filename);
for (int i = 0; i < N; i ++) {
for (int j = 0; j < N; j ++) {
if (j == N-1) {
out << C_n[i][j] << "\n";
} else {
out << C_n[i][j] << " ";
}
}
}
out.close();
}

double** create_matrix(int const& N){
double** mat = new double*[N];

for (int i = 0; i < N; i++) {
mat[i] = new double[N];
for (int j = 0; j < N; j++) {
mat[i][j] = 0.0;
}
}
return mat;
}

void initial_gaussian(double**& C_n, int const& N, double const& L) {
int i;
int j;
#pragma omp parallel for default(none) private(i, j) shared(C_n, N, L) schedule(guided) 
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
C_n[i][j] = exp(-(pow((i-N/2)*L/N, 2) + pow((j-N/2)*L/N, 2))/(L*L/8));
}
}
}

void apply_boundary_conditions(int const& N, double const& dt, double const& dx, double const& u, double const& v, double** const& C_n, double**& C_n_1) {
double C_n_up;
double C_n_down;
double C_n_left;
double C_n_right;
int i;
int j;
#pragma omp parallel for default(none) private(i, j, C_n_up, C_n_down, C_n_left, C_n_right) shared(C_n, C_n_1, N, dt, dx, u, v) schedule(guided)
for (i = 0; i < N; i ++) {
for (j = 0; j < N; j ++) {
if (i != 0 && i != N-1) {
C_n_up = C_n[i-1][j];
C_n_down = C_n[i+1][j];
if (j != 0 && j != N-1) {
C_n_left = C_n[i][j-1];
C_n_right = C_n[i][j+1];
}
else if (j == 0) {
C_n_left = C_n[i][N-1];
C_n_right = C_n[i][j+1];
} 
else {
C_n_left = C_n[i][j-1];
C_n_right = C_n[i][0];
}
}
else if (i == 0) {
C_n_up = C_n[N-1][j];
C_n_down = C_n[i+1][j];
if (j != 0 && j != N-1) {
C_n_left = C_n[i][j-1];
C_n_right = C_n[i][j+1];
}
else if (j == 0) {
C_n_left = C_n[i][N-1];
C_n_right = C_n[i][j+1];
} 
else {
C_n_left = C_n[i][j-1];
C_n_right = C_n[i][0];
}
} 
else {
C_n_up = C_n[i-1][j];
C_n_down = C_n[0][j];
if (j != 0 && j != N-1) {
C_n_left = C_n[i][j-1];
C_n_right = C_n[i][j+1];
}
else if (j == 0) {
C_n_left = C_n[i][N-1];
C_n_right = C_n[i][j+1];
} 
else {
C_n_left = C_n[i][j-1];
C_n_right = C_n[i][0];
}
}

C_n_1[i][j] = 0.25*(C_n_up + C_n_down + C_n_left + C_n_right) - (dt/(2*dx))*(u*(C_n_down - C_n_up) + v*(C_n_right - C_n_left));
}
}
}

void advection_simulation(int N, int NT, double L, double T, double u, double v) {

double** C_n = create_matrix(N);
double** C_n_1 = create_matrix(N);

double dx = L/N;
double dt = T/NT;

double best_grind_rate = 0.0;
double avg_grind_rate = 0.0;
double ts;
double te;

assert(dt<=dx/sqrt(2*(u*u + v*v)));

initial_gaussian(C_n, N, L);
apply_boundary_conditions(N, dt, dx, u, v, C_n, C_n_1);
swap(C_n, C_n_1);


for (int n = 0; n < NT; n ++) {
ts = omp_get_wtime();
apply_boundary_conditions(N, dt, dx, u, v, C_n, C_n_1);
swap(C_n, C_n_1);
te = omp_get_wtime(); 
cout << "Iteration: " << n << " - Grind Rate: " << floor(1/(te-ts)) << " iter/sec" << endl;
best_grind_rate = max(best_grind_rate, floor(1/(te-ts)));
avg_grind_rate += floor(1/(te-ts));

if (n%100 == 0){
char filename[100];
cout << "Writing output to file..." << endl;
int dummy_var = sprintf(filename, "./visualizing/outputs/output_%d.txt", n);
write_to_file(C_n, N, filename);
cout << "Output written to file!" << "\n" << endl;

}
}


cout << "Best grind rate: " << best_grind_rate << " iter/sec" << endl;
cout << "Average grind rate: " << floor(avg_grind_rate/NT) << " iter/sec" << endl;
}

int main(int argc, char** argv) {
int N = stoi(argv[1]);
int NT = stoi(argv[2]);
double L = stod(argv[3]);
double T = stod(argv[4]);
double u = stod(argv[5]);
double v = stod(argv[6]);

double ts;
double tf;
int num_threads;

cout << "Simulation Parameters:" << endl;
cout << "N = " << N << endl;
cout << "NT = " << NT << endl;
cout << "L = " << L << endl;
cout << "T = " << T << endl;
cout << "u = " << u << endl;
cout << "v = " << v << "\n" << endl;

cout << "Estimated memeory usage = " << N*N*sizeof(double)/1e6 << " MB" << "\n" << endl;

num_threads = 8;
cout << "Number of threads = " << num_threads << "\n" << endl;
omp_set_num_threads(num_threads);

cout << "Simulating..." << endl;
ts = omp_get_wtime();
advection_simulation(N, NT, L, T, u, v);
tf = omp_get_wtime();
cout << "Simulation Complete!" << endl;

cout << "Time taken = " << tf-ts << " seconds" << "\n" << endl;

return 0;

}