#include<iostream>
#include<omp.h>
using std::cout;
using std::endl;
int main()
{
int N = 10;
int * x = new int[N];
double sum = 0.0;	
#pragma omp parallel shared(x)
{
#pragma omp for schedule(static,5)
for(int i = 0; i < N; i++)
{
#ifdef _OPENMP
cout << "Thread id = " << omp_get_thread_num() 
<< " itr = " << i << endl;
#endif 
x[i] = i;
}
#pragma omp barrier
#pragma omp for reduction(+:sum)
for(int i = 0; i < N; i++)
{
sum = sum + x[i];
}
#pragma omp atomic
sum += 1;	
}
for(int i = 0; i < N; i++)
cout << x[i] << endl;
cout << "sum = " << sum << endl;
return 0;
}
