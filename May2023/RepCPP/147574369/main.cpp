#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

constexpr int THREAD_NUMBER = 12;

int main(void)
{
int id = 0;

omp_set_num_threads(THREAD_NUMBER);
#pragma omp parallel private(id)
{
id = omp_get_thread_num();
cout << "Thread: " << id << endl;
}

std::vector<int> test;
test.push_back(12);

#pragma omp parallel for
for (int i = 0; i< 10; i++) {
test.push_back(10+i);
}

#pragma omp parallel for
for (size_t i = 0 ; i < test.size(); i++) {
cout << "num " << test[i];
}

return 0;
}
