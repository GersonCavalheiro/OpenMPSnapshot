int
foo (int n)
{
int a[10][10];
int x, y;
#pragma omp parallel for collapse(2)	
for (x = 0; x < n; x++)		
for (y = 0; y < n; y++)
a[x][y] = x + y * y;
return a[0][0];
}
