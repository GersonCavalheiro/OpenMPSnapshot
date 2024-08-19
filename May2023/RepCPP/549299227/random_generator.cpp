
#include <iostream>
#include <random>
#include <omp.h>
using namespace std;


void worker(int &sum, int &turn)
{
int current_thread = omp_get_thread_num();
default_random_engine generator{static_cast<unsigned int>(current_thread * 10)};
uniform_int_distribution<int> distribution{ -50, 100 };

while (sum <= 1000)
{
#pragma omp critical
if (turn == current_thread && sum <= 1000)
{
int number = distribution(generator);
sum += number;
cout << "Thread " << current_thread << " generated " << number << ", "
<< "Sum: " << sum << endl;

if (sum >= 1000)
{
cout << "Thread " << current_thread << " wins!" << endl;
}

turn = (turn + 1) % omp_get_num_threads();
}
}
}


int main(int argc, char *argv[])
{
int thread_count = stoi(argv[1]);

int sum = 0, turn = 0;

#pragma omp parallel num_threads(thread_count)
worker(sum, turn);

return 0;
}
