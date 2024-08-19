#include <stdlib.h>
int main(int argc, char* argv[])
{ 
int i; 
int tmp;
tmp = 10;
int len=100;
if (argc>1)
len = atoi(argv[1]);
int a[len];
#pragma omp parallel for
for (i=0;i<len;i++)
{ 
a[i] = tmp;
tmp =a[i]+i;
}     
return 0;      
}
