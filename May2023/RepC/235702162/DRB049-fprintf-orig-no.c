#include <stdio.h>
int main(int argc, char * argv[])
{
int i;
int ret;
FILE * pfile;
int len = 1000;
int A[1000];
int _ret_val_0;
#pragma cetus private(i) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<len; i ++ )
{
A[i]=i;
}
pfile=fopen("mytempfile.txt", "a+");
if (pfile==((void * )0))
{
fprintf(stderr, "Error in fopen()\n");
}
#pragma cetus private(i) 
#pragma loop name main#1 
for (i=0; i<len;  ++ i)
{
fprintf(pfile, "%d\n", A[i]);
}
fclose(pfile);
ret=remove("mytempfile.txt");
if (ret!=0)
{
fprintf(stderr, "Error: unable to delete mytempfile.txt\n");
}
_ret_val_0=0;
return _ret_val_0;
}
