#include <omp.h>
#include <queue>
#include <iostream>
#include <cstdlib>
#include "stdio.h"
#include "stdlib.h"

int main(int argc, char **argv) {
const int size = (argc > 1 ? atoi(argv[1]) : 10);
omp_lock_t queue_lock, io_lock;
omp_init_lock(&queue_lock);
omp_init_lock(&io_lock);
std::queue<int> queue;
#pragma omp parallel sections shared(size, queue)
{
#pragma omp section
{
for (int i = 0; i < size; i++) {
omp_set_lock(&io_lock);
omp_set_lock(&queue_lock);

const int value = i;
queue.push(value);
std::cout << "Put " << value << " into queue"
<< std::endl;

omp_unset_lock(&io_lock);
omp_unset_lock(&queue_lock);

}
}
#pragma omp section
{
bool working = true;
while (working) {
omp_set_lock(&io_lock);
omp_set_lock(&queue_lock);

if (!queue.empty()) {
const int value = queue.front();
queue.pop();
std::cout << "Get " << value << " from queue"
<< std::endl;
working = (value < size - 1);
}

omp_unset_lock(&io_lock);
omp_unset_lock(&queue_lock);

}
}
}

omp_destroy_lock(&queue_lock);
omp_destroy_lock(&io_lock);
return 0;
}