#include<stdio.h>
#include<omp.h>
int main()
{
int choice;
printf("Enter 1 for parallel computing else anything for serial computing: ");
scanf("%d",&choice);
#pragma omp parallel if (choice==1)
if (omp_in_parallel())
{
#pragma omp single
printf("Parallel\n");
}
else
printf("Serial\n");
return 0;
}
