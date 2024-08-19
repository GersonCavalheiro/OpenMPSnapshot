#include <stdlib.h>
void setup(int N)
{
double * m_pdv_sum = (double * )malloc(sizeof (double)*N);
double * m_nvol = (double * )malloc(sizeof (double)*N);
{
int i = 0;
#pragma loop name setup#0 
for (; i<N;  ++ i)
{
m_pdv_sum[i]=0.0;
m_nvol[i]=(i*2.5);
}
}
{
int i = 0;
#pragma loop name setup#1 
for (; i<N;  ++ i)
{
printf("%lf\n", m_pdv_sum[i]);
printf("%lf\n", m_nvol[i]);
}
}
free(m_pdv_sum);
free(m_nvol);
return ;
}
int main()
{
int N = 1000;
int _ret_val_0;
setup(N);
return _ret_val_0;
}
