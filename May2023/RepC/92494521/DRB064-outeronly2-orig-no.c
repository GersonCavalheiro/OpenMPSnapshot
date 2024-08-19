int n=100, m=100;
double b[100][100];
void foo()
{
int i,j;
#pragma omp parallel for private(j)
for (i=0;i<n;i++)
for (j=1;j<m;j++) 
b[i][j]=b[i][j-1];
}
int main()
{
foo();
return 0;
}
