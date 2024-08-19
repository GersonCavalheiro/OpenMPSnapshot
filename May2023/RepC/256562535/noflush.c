#include <omp.h>
#include <stdio.h> 
#include <stdlib.h> 
int main() {
int data, flag = 0;
#pragma omp parallel num_threads(2)
{
if (omp_get_thread_num()==0) {
data = 42;
flag = 1;
}
else if (omp_get_thread_num()==1) {
while (flag < 1) { 
}
printf("flag=%d data=%d\n", flag, data);
}
}
return 0;
}
