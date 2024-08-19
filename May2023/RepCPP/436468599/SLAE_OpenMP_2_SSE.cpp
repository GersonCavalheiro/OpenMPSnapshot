#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <immintrin.h>

#pragma omp declare reduction(sseSum: __m128d: omp_out += omp_in) initializer (omp_priv = _mm_setzero_pd())

#define TYPE guided
#define CHUNK chunkSize
int matrixSize = 1;
int threadCount = 1;
int chunkSize = 1;

using namespace std::chrono;

double dotProduct(const double *vec1, const double *vec2, int size) {
__m128d vA = _mm_setzero_pd();
for (int i = 0; i < size; i += 2) {
vA += _mm_loadu_pd(&vec1[i]) * _mm_loadu_pd(&vec2[i]);
}
return _mm_hadd_pd(vA, vA)[0];
}

void printMat(double *mat, int rows, int columns, std::ostream &stream) {
for (int i = 0; i < rows; ++i) {
for (int j = 0; j < columns; ++j) {
stream << mat[i * columns + j] << " ";
}
stream << std::endl;
}
}

double *solveSLAE(const double *A, double *b, int N) {
auto *solution = new double[N]; 
std::fill(solution, solution + N, 0);
auto *prevSolution = new double[N]; 
std::fill(prevSolution, prevSolution + N, 0);

auto *Atmp = new double[N];
auto *r = new double[N];
auto *z = new double[N];
auto *rNext = new double[N];
auto *zNext = new double[N];
auto *alphaZ = new double[N];
auto *betaZ = new double[N];

double alpha;
double beta;

const double EPSILON = 1e-007;

double normb = sqrt(dotProduct(b, b, N));
double dotRR;

__m128d vA = _mm_setzero_pd();
__m128d vB = _mm_setzero_pd();

double res = 1;
double prevRes = 1;
bool diverge = false;
int divergeCount = 0;
int rightAnswerRepeat = 0;
int iterCount = 1;
#pragma omp parallel num_threads(threadCount) firstprivate(rightAnswerRepeat, divergeCount, diverge)
while (res > EPSILON || rightAnswerRepeat < 5) {
#pragma omp single
{
if (res < EPSILON) {
++rightAnswerRepeat;
} else {
rightAnswerRepeat = 0;
}
}

#pragma omp for schedule(TYPE, CHUNK)
for (int i = 0; i < N; i++) {
__m128d sum = _mm_setzero_pd();
for (int j = 0; j < N; j += 2) {
sum += _mm_loadu_pd(&A[i * N + j]) * _mm_loadu_pd(&prevSolution[j]);
}
Atmp[i] = _mm_hadd_pd(sum, sum)[0];
}
#pragma omp for schedule(TYPE, CHUNK)
for (int i = 0; i < N; i += 2) {
_mm_storeu_pd(&r[i], _mm_loadu_pd(&b[i]) - _mm_loadu_pd(&Atmp[i]));
_mm_storeu_pd(&z[i], _mm_loadu_pd(&r[i]));
}
#pragma omp for schedule(TYPE, CHUNK)
for (int i = 0; i < N; i++) {
__m128d sum = _mm_setzero_pd();
for (int j = 0; j < N; j += 2) {
sum += _mm_loadu_pd(&A[i * N + j]) * _mm_loadu_pd(&z[j]);
}
Atmp[i] = _mm_hadd_pd(sum, sum)[0];
}
#pragma omp single
{
vA = _mm_setzero_pd();
}
#pragma omp for schedule(TYPE, CHUNK) reduction(sseSum: vA, vB)
for (int i = 0; i < N; i += 2) {
vA += _mm_loadu_pd(&r[i]) * _mm_loadu_pd(&r[i]);
vB += _mm_loadu_pd(&Atmp[i]) * _mm_loadu_pd(&z[i]);
}
#pragma omp single
{
dotRR = _mm_hadd_pd(vA, vA)[0];
alpha = dotRR / _mm_hadd_pd(vB, vB)[0];
}
#pragma omp for schedule(TYPE, CHUNK)
for (int i = 0; i < N; i += 2) {
_mm_storeu_pd(&alphaZ[i], _mm_loadu_pd(&z[i]) * _mm_set1_pd(alpha));
_mm_storeu_pd(&solution[i], _mm_loadu_pd(&prevSolution[i]) + _mm_loadu_pd(&alphaZ[i]));
}
#pragma omp for schedule(TYPE, CHUNK)
for (int i = 0; i < N; i++) {
__m128d sum = _mm_setzero_pd();
for (int j = 0; j < N; j += 2) {
sum += _mm_loadu_pd(&A[i * N + j]) * _mm_loadu_pd(&alphaZ[j]);
}
Atmp[i] = _mm_hadd_pd(sum, sum)[0];
}
#pragma omp single
{
vA = _mm_setzero_pd();
}
#pragma omp for schedule(TYPE, CHUNK) reduction(sseSum: vA)
for (int i = 0; i < N; i += 2) {
_mm_storeu_pd(&rNext[i], _mm_loadu_pd(&r[i]) - _mm_loadu_pd(&Atmp[i]));
vA += _mm_loadu_pd(&rNext[i]) * _mm_loadu_pd(&rNext[i]);
}
#pragma omp single
{
beta = _mm_hadd_pd(vA, vA)[0] / dotRR;
}
#pragma omp for schedule(TYPE, CHUNK)
for (int i = 0; i < N; i += 2) {
_mm_storeu_pd(&betaZ[i], _mm_loadu_pd(&z[i]) * _mm_set1_pd(beta));
_mm_storeu_pd(&zNext[i], _mm_loadu_pd(&rNext[i]) + _mm_loadu_pd(&betaZ[i]));
}
#pragma omp single
{
res = sqrt(_mm_hadd_pd(vA, vA)[0]) / normb;
}
if (prevRes < res || res == INFINITY || res == NAN) {
#pragma omp single
{
++divergeCount;
}
if (divergeCount > 10 || res == INFINITY || res == NAN) {
diverge = true;
break;
}
} else {
#pragma omp single
{
divergeCount = 0;
}
}
#pragma omp single
{
prevRes = res;
++iterCount;
}
#pragma omp for schedule(TYPE, CHUNK)
for (long i = 0; i < N; i += 2) {
_mm_storeu_pd(&prevSolution[i], _mm_loadu_pd(&solution[i]));
_mm_storeu_pd(&r[i], _mm_loadu_pd(&rNext[i]));
_mm_storeu_pd(&z[i], _mm_loadu_pd(&zNext[i]));
}
}
delete[](prevSolution);
delete[](Atmp);
delete[](r);
delete[](z);
delete[](rNext);
delete[](zNext);
delete[](alphaZ);
delete[](betaZ);

std::cout << "iterCount: " << iterCount << std::endl;

if (diverge) {
delete[](solution);
return nullptr;
} else {
return solution;
}
}

int main(int argc, char *argv[]) {
if (argc != 5) {
std::cout << "Program needs 2 arguments: size, threadCount, filename, chunkPercent" << std::endl;
return 0;
}
int N = atoi(argv[1]);
matrixSize = N;
threadCount = atoi(argv[2]);
chunkSize = matrixSize / threadCount * atof(argv[4]);
std::cout << "chunkSize: " << chunkSize << std::endl;

const std::string &fileName = argv[3];
std::ofstream fileStream(fileName);
if (!fileStream) {
std::cout << "error with output file" << std::endl;
return 0;
}

fileStream << "Matrix size: " << N << " thread num: " << threadCount << std::endl;

auto *b = new double[N];
auto *u = new double[N];
auto *A = new double[N * N];

for (int i = 0; i < N; i++) {
for (int j = 0; j < N; j++) {
if (i == j) {
A[i * N + j] = 2;
} else {
A[i * N + j] = 1;
}
}
u[i] = sin(2 * M_PI * i / double(N));
}
for (int i = 0; i < N; i++) {
__m128d vA = _mm_setzero_pd();
for (int j = 0; j < N; j += 2) {
vA += _mm_loadu_pd(&A[i * N + j]) * _mm_loadu_pd(&u[j]);
}
b[i] = _mm_hadd_pd(vA, vA)[0];
}

auto startTime = system_clock::now();
double *solution = solveSLAE(A, b, N);
auto endTime = system_clock::now();
auto duration = duration_cast<nanoseconds>(endTime - startTime);

if (solution != nullptr) {
fileStream << "Answer:" << std::endl;
printMat(u, 1, N, fileStream);
fileStream << "SLAE solution:" << std::endl;
printMat(solution, 1, N, fileStream);
fileStream << "Time: " << duration.count() / double(1000000000) << "sec" << std::endl;
std::cout << "Time: " << duration.count() / double(1000000000) << "sec" << std::endl;
} else {
fileStream << "Does not converge" << std::endl;
}

delete[](solution);
delete[](b);
delete[](u);
delete[](A);
return 0;
}
