#include <stdio.h>
#include <time.h>
#include <omp.h>
int main()
{
double x;
long int i,j,n;
int flag,thrd,choice;
time_t begin = time(NULL); 
printf("Enter your Choice :\n ");
printf("\tSerial Code(Enter 0) \n");
printf("\tParallel Code(Enter 1) \n");
scanf("%d",&choice);
printf("Enter Number till which you want to generate Prime Numbers \n");
scanf("%ld",&n);
if(choice==1){
printf("Enter Number of Threads \n ");
scanf("%d",&thrd);
omp_set_num_threads(thrd);
}
#pragma omp parallel private(j,flag) shared(thrd,i,n) if(choice){
if(omp_in_parallel){
int t = omp_get_thread_num(),my_end,my_start;
#pragma omp for schedule(dynamic,n/thrd)
for(i=2;i<n;i++){
flag=0;
for (j=2; j <= i/2 ;++j){
if(i%j==0){
flag=1;
break;
}
}
if(flag==0){	
printf(" %ld \n",i);
}
}
}
else{
for(i=2;i<n;i++){
flag=0;
for (j=2; j <= i/2 ;++j){
if(i%j==0){
flag=1;
break;
}
}
if(flag==0){		
printf(" %ld \n",i);
}
}
}
}
time_t end = time(NULL);
x= end - begin;
printf("\n %f seconds \n",x);
return 0;  
}
