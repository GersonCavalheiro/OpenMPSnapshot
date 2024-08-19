#define N 1000
double a[N][N],v[N],v_out[N];
void mv()
{           
int i,j;
for (i = 0; i < N; i++)
{         
float sum = 0.0;
#pragma omp parallel for reduction(+:sum)
for (j = 0; j < N; j++)
{ 
sum += a[i][j]*v[j];
}  
v_out[i] = sum;
}         
}
int main()
{
mv();
return 0;
}
