#define N 100
#define M 100 
#define K 100
double a[N][M],b[M][K],c[N][K];
int mmm()   
{           
int i,j,k;
#pragma omp parallel for private(j,k)
for (i = 0; i < N; i++) 
for (k = 0; k < K; k++) 
for (j = 0; j < M; j++)
c[i][j]= c[i][j]+a[i][k]*b[k][j];
return 0; 
} 
int main()
{
mmm();
return 0;
}  
