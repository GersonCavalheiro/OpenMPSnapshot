#include <iostream>
#include <armadillo>
#include <stdio.h>
#include <omp.h>

using namespace std;
using namespace arma;



int main(int argc, char const *argv[]) {

int   x_points, y_points;
float x_b     , y_b     ;
float x_e     , y_e     ;

stringstream convert0(argv[1]);
stringstream convert1(argv[2]);
stringstream convert2(argv[3]);
stringstream convert3(argv[4]);
stringstream convert4(argv[5]);
stringstream convert5(argv[6]);

convert0 >> x_points;
convert1 >> x_b;
convert2 >> x_e;
convert3 >> y_points;
convert4 >> y_b;
convert5 >> y_e;

cout << "X points: "    << x_points << "\n";
cout << "X beggining: " << x_b      << "\n";
cout << "X end: "       << x_e      << "\n";
cout << "Y points: "    << y_points << "\n";
cout << "Y beggining: " << y_b      << "\n";
cout << "X end: "       << x_e      << "\n";

mat A(x_points, y_points);
rowvec parameters(6);

float x_ofst = (x_e - x_b) / x_points;
float y_ofst = (y_e - y_b) / y_points;

parameters(0) = x_points; parameters(1) = x_ofst; parameters(2) = x_b;
parameters(3) = y_points; parameters(4) = y_ofst; parameters(5) = y_b;

float x_i = x_b;
float y_i = y_b;

int i, j;
#pragma omp parallel for default(none) shared(A, x_b, y_b, x_ofst, y_ofst, x_points, y_points) private(i, j, x_i, y_i)
for (i = 0; i < x_points; i++) { 
x_i = x_b + i * x_ofst;
y_i = y_b;
for (j = 0; j < y_points; j++) {
A(i, j) = 2. * x_i * x_i + y_i * y_i;
y_i += y_ofst;
}
}

parameters.save("data/outputs/pmts.dat", raw_ascii);
A.save("data/outputs/A.dat", raw_binary);

return 0;
}
