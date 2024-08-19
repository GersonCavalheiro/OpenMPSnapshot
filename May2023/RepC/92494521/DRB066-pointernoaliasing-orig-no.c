#include <stdlib.h>
void setup(int N)
{
double * m_pdv_sum = (double* ) malloc (sizeof (double) * N );
double * m_nvol = (double* ) malloc (sizeof (double) * N );
#pragma omp parallel for schedule(static)
for (int i=0; i < N; ++i ) 
{ 
m_pdv_sum[ i ] = 0.0;
m_nvol[ i ]   = i*2.5;
}
free(m_pdv_sum);
free(m_nvol);
}
int main()
{
int N =1000;
setup(N);
}
