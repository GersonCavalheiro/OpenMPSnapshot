int a[100][100];
int main()
{
int i,j;
#pragma omp parallel for private(j)
for (i=0;i<100;i++)
for (j=0;j<100;j++)
a[i][j]=a[i][j]+1;
return 0;
}
