#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <cmath>
#include <limits>

using namespace std;

using matrix = vector<vector<double>>;


double ExactSolution(double _x, double _y) {
return exp(1 - pow(_x, 2) - pow(_y, 2));
}

double mu1(double y)
{
return exp(-pow(y, 2));
}

double mu2(double y)
{
return exp(-pow(y, 2));

}

double mu3(double x)
{
return exp(-pow(x, 2));
}

double mu4(double x)
{
return exp(-pow(x, 2));
}

double Laplas(double _x, double _y)
{
return -4 * (1 - pow(_x, 2) - pow(_y, 2)) * exp(1 - pow(_x, 2) - pow(_y, 2));
}

matrix calcDiscreapancy(matrix& r, const matrix& V, double param_x, double param_y, double A, int a, int b, int c, int d, size_t n, size_t m) {
double h = (b - a) / (double)n;
double k = (d - c) / (double)m;
for (int i = 1; i < m; i++) {
for (int j = 1; j < n; j++) {
r[i - 1][j - 1] = V[i][j] * A + (V[i][j + 1] + V[i][j - 1]) / param_x + (V[i + 1][j] + V[i - 1][j]) / param_y + Laplas(a + j * h, c + i * k);
}
}
return r;
}

matrix calcAh(matrix& H, matrix& r, double A, double param_x, double param_y, size_t n, size_t m) {
matrix Ah;
Ah.assign(m, vector<double>(n));
for (int i = 1; i < r.size(); i++) {
for (int j = 1; j < r[0].size(); j++) {
Ah[i - 1][j - 1] = A * H[i - 1][j - 1] + (H[i - 1][j] + H[i - 1][j]) / param_x + (H[i][j - 1] + H[i][j - 1]) / param_y; 
}
}
return Ah;
}

double calcAhh(matrix& Ah, matrix& r) {
double temp = 0;
for (int i = 1; i < r.size(); i++) {
for (int j = 1; j < r[0].size(); j++) {
temp += Ah[i - 1][j - 1] * r[i - 1][j - 1];
}
}
return temp;
}


matrix calcH(matrix& H, matrix& r, double betta) {
for (int i = 1; i < H.size(); i++) {
for (int j = 1; j < H[0].size(); j++) {
H[i - 1][j - 1] = r[i - 1][j - 1] * (-1) + H[i - 1][j - 1] * betta; 
}
}
return  H;
}


double calcAlpha(matrix& H, matrix& r, double param_x, double param_y, double A, double& Ahh, size_t n, size_t m) {

double temp = 0, betta = 0, alpha = 0;
matrix Ah = calcAh(H, r, A, param_x, param_y, n, m);



temp = calcAhh(Ah, r); 

betta = temp / Ahh;
H = calcH(H, r, betta);


Ahh = 0; temp = 0;

Ah = calcAh(H, r, A, param_x, param_y, n, m);

Ahh = calcAhh(Ah, H);


temp = calcAhh(r, H);
alpha = -temp / Ahh;

return alpha;

}

double calcError(matrix V, double h, double k, int a, int c) {
double error = 0, max_error = 0;

for (int i = 0; i < V.size(); i++) {
for (int j = 0; j < V[0].size(); j++) {
error = abs(V[i][j] - ExactSolution(a + j * h, c + i * k));
if (error > max_error) {
max_error = error;
}
}
}

return max_error;
}
