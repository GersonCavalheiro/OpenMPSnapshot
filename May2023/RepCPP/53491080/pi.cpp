
#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <iomanip>
#include <cmath>
#define _USE_MATH_DEFINES

using namespace std;

#define SYSTEMTIME clock_t


void printTimeUntilNow(SYSTEMTIME time1){
char st[100];

sprintf(st, "Time: %3.3f seconds\n", (double)(clock() - time1) / CLOCKS_PER_SEC);
cout << st;
}

int main (int argc, char *argv[])
{

SYSTEMTIME time1;
double area, pi, x;
int i, n;
char c;

time1 = clock();

cout << "Numero de processadores: " << omp_get_num_procs() << endl;

cout << "Numero de divisoes ? "; 
cin >> n; 
area = 0.0;

#pragma omp parallel for private(x) reduction(+:area)
for (i = 0; i < n; i++) {
x = (i+0.5)/n;
area += 4.0/(1.0 + x*x);
}
pi = area / n;



printTimeUntilNow(time1);

cout << setprecision(18) << "PI       = " << pi << endl << endl;
cout << setprecision(18) << "R.PI     = " << M_PI << endl << endl;

cout << setprecision(18) << "DIFF     = " << abs(pi - M_PI) << endl;
cout << setprecision(18) << "PRECISION= " << abs(pi)/(double)M_PI << endl;


cout << "Enter para continuar ...";
cin.get(c);
cin.get(c);
}
