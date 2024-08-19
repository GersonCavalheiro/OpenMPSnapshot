#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main()
{
int a[10], b[10], c[10],i,t[10];
for(i=0;i<10;i++){
a[i] = i+1;
b[i] = i+2;
}
#pragma omp parallel for shared(a, b, c) private(i) schedule(static, 1)
for (i=0;i<10;i++){
c[i]=a[i]+b[i];
t[i]=omp_get_thread_num();
printf("[+] Thread %d works on index%d\n", omp_get_thread_num(), i);
}
for(i=0;i<10;i++){
printf("[~] %d found by thread %d\n",c[i],t[i]);	
}
return 0;
}
*************************************************************************
root@kali:~/opt/pdc# gcc vectoradd.c -fopenmp
root@kali:~/opt/pdc# ./a.out
[+] Thread 0 works on index0
[+] Thread 0 works on index4
[+] Thread 0 works on index8
[+] Thread 3 works on index3
[+] Thread 3 works on index7
[+] Thread 2 works on index2
[+] Thread 2 works on index6
[+] Thread 1 works on index1
[+] Thread 1 works on index5
[+] Thread 1 works on index9
[~] 3 found by thread 0
[~] 5 found by thread 1
[~] 7 found by thread 2
[~] 9 found by thread 3
[~] 11 found by thread 0
[~] 13 found by thread 1
[~] 15 found by thread 2
[~] 17 found by thread 3
[~] 19 found by thread 0
[~] 21 found by thread 1
