#include <stdio.h>
int main() {
printf("Serial region.\n");
#pragma omp parallel for
for (int i = 0; i < 20; i++) {
printf("Hello %d\n", i);
}
printf("Serial region.\n");
}
