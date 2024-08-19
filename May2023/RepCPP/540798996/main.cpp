#include <iostream>

#include <omp.h>
int main () {

#pragma omp parallel
{
printf("Hello World !\n");
}
}
