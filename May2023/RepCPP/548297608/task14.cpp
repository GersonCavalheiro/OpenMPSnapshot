
#include <iostream>

int main() {
int N = 210;
int result = 0;

#pragma omp parallel for schedule(guided, 4) reduction(+: result)
for (int i = 0; i < N; i++) {
result += i + i + 1;
}

std::cout << std::to_string(N) + "^2 = " + std::to_string(result) + "\n";

return 0;
}
