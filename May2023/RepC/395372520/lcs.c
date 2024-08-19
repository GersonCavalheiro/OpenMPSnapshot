#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
char *readFile(char *filename, int *size)
{
char *buffer = NULL;
*size = 0;
FILE *fp = fopen(filename, "r");
fseek(fp, 0, SEEK_END); 
*size = ftell(fp);      
rewind(fp);
buffer = malloc((*size + 1) * sizeof(*buffer)); 
int err = fread(buffer, *size, 1, fp); 
buffer[*size] = '\0';
return (buffer);
}
int max(int a, int b)
{
return (a > b) ? a : b;
}
int LCSubStr(char *x, char *y, int m, int n)
{
int **LCSuff = (int **)malloc((m + 1) * sizeof(int *));
for (int i = 0; i < m + 1; i++)
LCSuff[i] = (int *)malloc((n + 1) * sizeof(int));
int result = 0; 
#pragma omp target map(tofrom:result) map(tofrom:LCSuff[0:m+1])
#pragma omp teams distribute parallel for reduction(max:result) collapse(2) schedule(guided)
for (int i = 0; i <= m; i++)
{
for (int j = 0; j <= n; j++)
{
if (i == 0 || j == 0)
LCSuff[i][j] = 0;
else if (x[i - 1] == y[j - 1])
{
LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1;
result = max(result, LCSuff[i][j]);
}
else
LCSuff[i][j] = 0;
}
}
return result;
}
int main()
{
int m, n;
char *x = readFile("seqA.txt", &m);
char *y = readFile("seqB.txt", &n);
printf("\nLength of Longest Common Substring is %d\n", LCSubStr(x, y, m, n));
return 0;
}