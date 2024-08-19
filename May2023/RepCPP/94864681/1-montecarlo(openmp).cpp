#include <iostream>
#include <ctime>
#include <math.h>
#include <omp.h>

using namespace std;
double a,b;
double I,S,x,y,f;
int K;
const int N = 100*1024*1024;

double func(double xx)
{
double ff = xx*xx + 3;
return ff;
}

void main()
{
cout<<"Input a = "; cin>>a;
clock_t time = clock();
b = func(a);
K = 0;

omp_set_num_threads(2);

srand(time);
#pragma omp parallel for default(none) shared(N) reduction(+: K)
for(int i=0; i<=N; i++)
{
x = a*abs(double(rand())/RAND_MAX);
y = b*abs(double(rand())/RAND_MAX);
f = func(x);
if (y<=f) K++;
}
S = a * b;
I = K*S/N;
cout<<"Integral = "<<I;

time = clock() - time;
cout<<" time = "<<time<<" msc\n";

int q;
cin>>q;

}
