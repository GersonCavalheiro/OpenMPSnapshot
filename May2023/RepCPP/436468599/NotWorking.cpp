#include <iostream>
#include <omp.h>
#include <random>

void doWork(const double *workArray, double *actualWork, int workerNum, double workSpeed, int anotherWorker) {
double work = workArray[anotherWorker] - actualWork[anotherWorker];
while (work > 0) {
#pragma omp critical
{
work -= workSpeed;
actualWork[workerNum] += workSpeed;
}
}
}

int main() {
int workerCount = 3;
int iterCount = 5;
auto *workArray = new double[workerCount];
auto *actualWork = new double[workerCount];
auto *totalWork = new double[workerCount];

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(1000, 5000);

#pragma omp parallel default(none) shared(iterCount, workerCount, workArray, actualWork, totalWork, std::cout, dis, gen) num_threads(workerCount)
{
int workerNum = omp_get_thread_num();
double workSpeed = 0.00001;
for (int i = 0; i < iterCount; ++i) {
workArray[workerNum] = double(dis(gen)) / 1000;
actualWork[workerNum] = 0;

#pragma omp barrier
doWork(workArray, actualWork, workerNum, workSpeed, workerNum);
for (int k = 0; k < workerCount; ++k) {
if (actualWork[k] > workArray[k] && k != workerNum) {
doWork(workArray, actualWork, workerNum, workSpeed, k);
}
}
totalWork[workerNum] += actualWork[workerNum];

#pragma omp single
{
std::cout << "Iter" << i << ":   ";
for (int j = 0; j < workerCount; ++j) {
std::cout << workArray[j] << "|" << actualWork[j] << "   ";
}
std::cout << std::endl;
}
}
}

double max = 0;
int indexOfMax = 0;
#pragma omp parallel for default(none) shared(workerCount, totalWork, max, indexOfMax)
for (int k = 0; k < workerCount; ++k) {
if (totalWork[k] > max) {
max = totalWork[k];
indexOfMax = k;
}
}

std::cout << "WINNER: worker" << indexOfMax << " (" << max << ")" << std::endl;

delete[] (workArray);
delete[] (actualWork);
delete[] (totalWork);
return 0;
}