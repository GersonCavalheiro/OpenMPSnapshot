#include <stdio.h>
#include <omp.h>
int main()
{
int stack[] = {1, 2, 3, 4, 5};
int top = 5;
int pop;
printf("\nStack Initial Declaration:\n");
for (int i = 0; i < top; i++)
{
printf("%d ", stack[i]);
}
printf("\n");
#pragma omp parallel shared(stack, top) private(pop)
{
#pragma omp critical
pop = stack[top];
top = top - 1;
}
printf("\nStack after construct: \n");
for (int i = 0; i < top; i++)
{
printf("%d ", stack[i]);
}
return 0;
}
