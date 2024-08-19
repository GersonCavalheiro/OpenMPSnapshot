#include <iostream>
#include <omp.h>
using namespace std;
int main()
{
int n = 100, i;
int a[n], b[n];
#pragma omp parallel num_threads(4)
{
#pragma omp sections private(i)
{
#pragma omp section
{
for (i=0; i<n; i++)
{
a[i] = 1;
}
}
#pragma omp section
{
for (i=0; i<n; i++)
{
b[i] = 2;
}
}
}
}
for (i=0; i<n;i++)
cout << a[i] << " ";
cout << "\n";
for (i=0; i<n;i++)
cout << b[i] << " ";
cout << "\n";
return 0;
}
