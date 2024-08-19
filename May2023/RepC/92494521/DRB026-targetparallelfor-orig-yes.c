int main(int argc, char* argv[])
{
int i;
int len = 1000;
int a[1000];
for (i=0; i<len; i++)
a[i]= i;
#pragma omp target map(a[0:len])
#pragma omp parallel for
for (i=0;i< len -1 ;i++)
a[i]=a[i+1]+1;
return 0;
}
