#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
int compare_exchange(int *a, int *b) {
int tmp;
if (*a > *b) {
tmp = *a;
*a = *b;
*b = tmp;
return 1;
} else {
return 0;
}
}
void bubble_sort(int n, int *a){
int flag;
for (int i = n; i > 1; i--) {
flag = 0;
for (int j = 0; j < i; j++) {
if (compare_exchange(&a[j], &a[j+1])) {
flag = 1;
}
}
if (!flag) break;
}
}
void para_oddeven_sort(int n, int* a) {
for (size_t i = 0; i < n; i++) {
if (i%2) {
#pragma omp parallel for
for (int j = 0; j < n/2; j++) {
compare_exchange(&a[2*j+1], &a[2*j+2]);
}
} else {
#pragma omp parallel for
for (int j = 0; j < n/2; j++) {
compare_exchange(&a[2*j], &a[2*j+1]);
}
}
}
printf("Sorted odd/even:\n");
for (int i = 0; i < n; i++) {
printf("%d\n", a[i]);
}
}
int main(int argc, char const *argv[]) {
int n = 10;
int a[10] = {2,1,6,14,25,16,27,8,10,9};
bubble_sort(n, a);
printf("Sorted:\n");
for (int i = 0; i < n; i++) {
printf("%d\n", a[i]);
}
return 0;
}
