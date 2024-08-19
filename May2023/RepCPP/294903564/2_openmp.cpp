#include <iostream>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <cstdlib>
#include <omp.h>

using namespace std;

double f(double x)
{
return 4 / (1 + x * x);
}

double I(double a, double b, int n, double y)
{
return ((b - a) / (2 * n) * y);
}


int main() {

int n; double a, b, y, dy, In;

a = 0.0;
b = 1.0;
n = 1000000000; 

if (n > 1)
{

y = 0.;
dy = (b - a) / n;
y += f(a) + f(b);
#pragma omp parallel for reduction(+:y) num_threads(8)
for (int i = 1; i < n; i++) 
{
y += 2 * (f(a + dy * i));
}

In = I(a, b, n, y);
cout << In;
}

else {cout << "Wrong data";}
}