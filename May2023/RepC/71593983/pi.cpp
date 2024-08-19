#include<iostream>
#include<omp.h>
using std::cout;
using std::endl;
int main()
{
int n = 1000;
double factor = 1.0;
double sum = 0.0;
#pragma omp parallel for reduction(+:sum) private(factor)
for(int i = 0; i < n; i++)
{
if(i % 2 == 0)
factor = 1.0;
else
factor = -1.0;
sum += factor/(2*i+1);
}
cout << 4.0*sum << endl;
return 0;
} 
