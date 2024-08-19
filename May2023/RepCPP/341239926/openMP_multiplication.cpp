
#include <omp.h>

#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>

#define FILENAME "openMP.csv"

using namespace std;

class mxm_kernel;

void display_matrix(float *m, int matSize)
{
std::cout << std::endl
<< "=================" << std::endl;
for (int i = 0; i < matSize; i++)
{
std::cout << m[i] << " ";
if (i > 10)
break;
}
std::cout << std::endl
<< "=================" << std::endl;
}

inline int prevPowerOfTwo(int x)
{
if (x < 2)
{
return 0;
}
x |= x >> 1;
x |= x >> 2;
x |= x >> 4;
x |= x >> 8;
x |= x >> 16;
return x - (x >> 1);
}

template <typename T>
void local_mxm(T *MA, T *MB, T *MC, unsigned int matSize, unsigned int num_thread)
{
std::ofstream fileStream;

int blockSize = 512;

int i = 0, j = 0, k = 0, jj = 0, kk = 0;
float tmp;
int chunk = 1;
int tid;
auto start = std::chrono::steady_clock::now();
#pragma omp parallel num_threads(num_thread) shared(MA, MB, MC, matSize, chunk) private(i, j, k, jj, kk, tid, tmp)

{
tid = omp_get_thread_num();
if (tid == 0)
{
std::cout << "Number of threads optimal: " << omp_get_num_threads() << std::endl;
std::cout << "Number of threads real: " << num_thread << std::endl;
}

#pragma omp for schedule (static, chunk)
for (jj = 0; jj < matSize; jj += blockSize)
{
for (kk = 0; kk < matSize; kk += blockSize)
{
for (i = 0; i < matSize; i++)
{
for (k = kk; k < ((kk + blockSize) > matSize ? matSize : (kk + blockSize)); k++)
{

for (j = jj; j < ((jj + blockSize) > matSize ? matSize : (jj + blockSize)); j++)
{
MC[i * matSize +j] += MA[i * matSize +k] * MB[k * matSize +j];
}

}
}
}
}
}

auto end = std::chrono::steady_clock::now();
std::chrono::duration<double> time_seconds = end - start;
std::cout << "Time : " << time_seconds.count() << " seconds" << std::endl;
float flops = (2.0f * matSize * matSize * matSize / time_seconds.count()) * 1.0e-9f;
std::cout << "GFLOPs: " << flops << std::endl;
fileStream.open(FILENAME, fstream::app);
fileStream << matSize << ", " << time_seconds.count() << ", " << flops << "\n";
fileStream.close();
display_matrix(MC, matSize);
}

int main(int argc, char *argv[])
{
float *MA;
float *MB;
float *MC;

bool error = false;

int op;
unsigned int startSize, endSize, step;
int platformIndex = 1;
int deviceIndex = 1;
unsigned int number_thread;

std::cout << std::endl
<< "********** openMP Matrix Multiplication ****************" << std::endl
<< std::endl;
op = 1;
do
{
std::cout << "Selection?: 1 => Block Multiplication | 0 => Stop" << std::endl;
std::cin >> op;
if (op == 0)
{
std::cout << "Exiting" << std::endl;
return 0;
}

std::cout << "Number of threads? " << std::endl;
std::cin >> number_thread;

printf("Parameters?: [StartSize] [Step] [EndSize]\n ");
std::cin >> startSize >> step >> endSize;

int blockSize;

std::cout << "\n****Parallel Block Multiplication****" << std::endl;

do
{
MA = new float[startSize * startSize];
MB = new float[startSize * startSize];
MC = new float[startSize * startSize];

for (int i = 0; i < startSize; i++)
for (int j = 0; j < startSize; j++)
MA[i * startSize + j] = (double)1.0;

for (int i = 0; i < startSize; i++)
for (int j = 0; j < startSize; j++)
MB[i * startSize + j] = (double)(i + 1);

local_mxm(MA, MB, MC, startSize, number_thread);
startSize += step;

delete[] MA;
delete[] MB;
delete[] MC;
} while (startSize <= endSize);

} while (op != 0);

return 0;
}
