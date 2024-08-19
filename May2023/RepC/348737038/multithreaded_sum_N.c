#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main()
{
int s=0,i,n=2021;
#pragma omp parallel for 
for (i=1;i<=n;i++){
s=s+i;
printf("[+] Thread %d works on index%d\n", omp_get_thread_num(), i);
}
printf("[~] sum of 2021 numbers %d \n",s);	
return 0;
}
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main()
{
int s=0,i,n=2021;
#pragma omp parallel for 
for (i=1;i<=n;i++){
#pragma omp critical
s=s+i;
printf("[~] Thread %d works on index%d\n", omp_get_thread_num(), i);
}
printf("[+] sum of 2021 numbers %d \n",s);
return 0;
}
