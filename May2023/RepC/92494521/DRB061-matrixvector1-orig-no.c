#define N 100
double a[N][N],v[N],v_out[N];
int mv()
{           
int i,j;
#pragma omp parallel for private (i,j)
for (i = 0; i < N; i++)
{         
float sum = 0.0;
for (j = 0; j < N; j++)
{ 
sum += a[i][j]*v[j];
}  
v_out[i] = sum;
}         
return 0; 
}
int main()
{
mv();
return 0;
}
