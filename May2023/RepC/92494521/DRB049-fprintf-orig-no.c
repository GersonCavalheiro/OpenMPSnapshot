#include <stdio.h>
int main(int argc, char* argv[])
{
int i;
int ret;
FILE* pfile;
int len=1000;
int A[1000];
for (i=0; i<len; i++)
A[i]=i;
pfile = fopen("mytempfile.txt","a+");
if (pfile ==NULL)
{
fprintf(stderr,"Error in fopen()\n");
}
#pragma omp parallel for
for (i=0; i<len; ++i)
{
fprintf(pfile, "%d\n", A[i] );
}
fclose(pfile);
ret = remove("mytempfile.txt");
if (ret != 0)
{
fprintf(stderr, "Error: unable to delete mytempfile.txt\n");
}
return 0;
}
