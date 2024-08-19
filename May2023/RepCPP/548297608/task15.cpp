
#include <iostream>
#include <vector>

int main() {
int a, b;
std::cin >> a >> b;
bool took[b + 1];
for (int i = 0; i < b + 1; i++) {
took[i] = false;
}
std::vector<int> result;

for (int i = 2; i <= b; i++) {
if (!took[i]) {
if (i >= a) result.push_back(i);

#pragma omp parallel for
for (int j = 2 * i; j <= b; j += i) {
took[j] = true;
}
}
}

std::cout << "Result = [";
for (int i = 0; i < result.size(); i++) {
if (i < result.size() - 1) std::cout << std::to_string(result[i]) + ", ";
else std::cout << std::to_string(result[i]) + "]\n";
}

return 0;
}
