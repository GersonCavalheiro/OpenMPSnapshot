
#include<omp.h>


template <typename T, typename Op>
void omp_scan(int n, const T* in, T* out, Op op) 
{
int p = 1;


#pragma omp parallel
{
p = omp_get_num_threads();
}


int k = n/p;


#pragma omp parallel for schedule(auto)
for(int i=0; i<n; i=i+k)
{	
out[i] = in[i];
for(int j = i+1; j<(i+k) && j<n; j++)
{
out[j] = op(in[j], out[j-1]);
}
}


for(int i = (2*k)-1; i<n; i=i+k)
{
out[i] = op(out[i], out[i-k]);
}


#pragma omp parallel for schedule(auto)
for(int i = k-1; i<n; i=i+k)
{
int j = i+1;
while(j<n && j<(i+k))
{
out[j] = op(out[j], out[i]);
j++;
}
}

} 




