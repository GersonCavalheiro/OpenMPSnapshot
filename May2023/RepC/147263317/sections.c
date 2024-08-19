#include <omp.h>
#include <stdio.h>
void usonian() {
printf("hi\n");
}
void english() {
printf("oh, hello!\n");
}
void homer() {
printf("yello\n");
}
int main() {
int sum;
#pragma omp parallel
{
#pragma omp sections
{
#pragma omp section
{
usonian();
}
#pragma omp section
{
english();
}
#pragma omp section
{
homer();
}
}
}
return 0;
}
