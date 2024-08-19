#include <stdio.h>
#include <unistd.h>
void task1(){
int i=0;
while(1){
printf("task1 - %d\n", i);
i++;
sleep(5);
}
}
void task2(){
int i=0;
while(1){
printf("task2 - %d\n", i);
i++;
sleep(4);
}
}
int main(int argc, char *argv[]) 
{ 
#pragma omp parallel   
{     
#pragma omp single      
{ 
printf("A ");         
#pragma omp task 
{
task2();
}         
#pragma omp task 
{
task1();
} 
#pragma omp taskwait
printf("is fun to watch ");
}
} 
printf("\n");
return(0); 
} 