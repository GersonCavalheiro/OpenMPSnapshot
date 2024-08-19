#include <iostream>
#include <omp.h>
#include <chrono>
#include <fstream>

using std::cout;
using std::ofstream;

constexpr char nl{'\n'};
constexpr int a{1000};
constexpr int b{1000};
constexpr int c{1000};

int m1[a][b];
int m2[b][c];
int mr[a][c];

int main(int argc, char *argv[])
{
if (argc != 2)
{
cout << "Incorrect Usage." << nl << "Correct Usage: ./openmp <number_of_threads: int>" << nl;
}
else
{
for (int i = 0; i < a; i++)
{
for (int j = 0; j < b; j++)
{
m1[i][j] = i + j;
m2[i][j] = i - j;
}
}

int num_of_threads = atoi(argv[1]);

auto start = std::chrono::high_resolution_clock::now();

omp_set_num_threads(num_of_threads);

#pragma omp parallel for
for (int q = 0; q < a; q++)
{
for (int w = 0; w < c; w++)
{
for (int e = 0; e < b; e++)
{
mr[q][w] += m1[q][e] * m2[e][w];
}
}
}

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
std::cout << "Calculation done in " << duration.count() << " microseconds\n";

ofstream outfile("outputmp.txt");
for (int z = 0; z < a; z++)
{
for (int y = 0; y < c; y++)
{
outfile << mr[z][y] << " ";
}
outfile << "\n";
}
}
}

