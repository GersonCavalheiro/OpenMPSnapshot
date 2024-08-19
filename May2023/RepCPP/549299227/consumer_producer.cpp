
#include <iostream>
#include <random>
#include <queue>
#include <omp.h>
#include <unistd.h>
using namespace std;


void producer(queue<int> &queue, unsigned int capacity)
{
int current_thread = omp_get_thread_num();
default_random_engine generator{static_cast<unsigned int>(current_thread)};
uniform_int_distribution<int> distribution{ 1, 10 };

while (true)
{
while (queue.size() >= capacity); 

#pragma omp critical
if (queue.size() < capacity)
{
int message = distribution(generator);
queue.push(message);
cout << "Producer " << current_thread << " produced " << message << endl;
}
}
}


void consumer(queue<int> &queue, unsigned int capacity)
{
int current_thread = omp_get_thread_num();

while (true)
{
while (queue.size() == 0); 

int message = 0;
#pragma omp critical
if (queue.size() != 0)
{
message = queue.front();
queue.pop();
cout << "Consumer " << current_thread << " consumed " << message << endl;
}

sleep(message);
}
}


int main(int argc, char *argv[])
{
int capacity = stoi(argv[1]), num_producer = stoi(argv[2]),
num_consumer = stoi(argv[3]);

queue<int> queue;

#pragma omp parallel num_threads(num_producer + num_consumer) shared(queue)
{
int current_thread = omp_get_thread_num();
if (current_thread < num_producer)
producer(queue, capacity);
else
consumer(queue, capacity);
}
}