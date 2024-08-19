
#include <iostream>

int main() {
const int I = 30;
int a[I] = {1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1};
int x = 0;

#pragma omp parallel for schedule(guided, 4) reduction(+: x)
for (int i = 0; i < I; i++) {
if (a[i] == 1) x += 1 << (I - i - 1);
}

std::cout << "x = " + std::to_string(x) + "\n";

return 0;
}
