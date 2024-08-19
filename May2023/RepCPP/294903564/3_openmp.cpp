#include <iostream>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <cstdlib>
#include <omp.h>

using namespace std;

double func(double x)

{

return pow(x, 3) - 3*sin(x);

}

int main()

{

double a=0, b=3;

double x, y;

int n;

cout << "n = ";

n = 100000000;

y=func(a);

#pragma omp parallel for reduction(+:y) num_threads(8)
for (int i=1; i<=n; i++)

{

x=a+i*(b-a)/(n+1);

if(func(x)<y)

y=func(x);

}

cout << endl;

cout << "e = " << (b-a)/(n+1) << endl;

cout << "x = " << x << endl;

cout << "y = " << y << endl;

return 0;

}