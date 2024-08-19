int main()
{
int i,j;
int n=100, m=100;
double b[n][m];
for(i=0;i<n; i++) 
for(j=0;j<n; j++) 
b[i][j]=(double)(i*j);
for (i=1;i<n;i++)
#pragma omp parallel for
for (j=1;j<m;j++)
b[i][j]=b[i-1][j-1];
return 0;
}
