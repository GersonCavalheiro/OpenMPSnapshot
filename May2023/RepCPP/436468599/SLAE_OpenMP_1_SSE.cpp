#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <immintrin.h>

#pragma omp declare reduction(sseSum: __m128d: omp_out += omp_in) initializer (omp_priv = _mm_setzero_pd())

#define TYPE guided
#define CHUNK chunkSize
#define BASE_CLAUSE default(none) num_threads(threadCount) schedule(TYPE, CHUNK)
int matrixSize = 1;
int threadCount = 1;
int chunkSize = 1;

using namespace std::chrono;

void matVecMul(const double *mat, const double *vec, int N, double *newVec) {
#pragma omp parallel for shared(mat, vec, N, newVec, chunkSize) BASE_CLAUSE
for (int i = 0; i < N; i++) {
__m128d vA = _mm_setzero_pd();
for (int j = 0; j < N; j += 2) {
vA += _mm_loadu_pd(&mat[i * N + j]) * _mm_loadu_pd(&vec[j]);
}
newVec[i] = _mm_hadd_pd(vA, vA)[0];
}
}

void mulByConst(const double *vec, double c, int size, double *newVec) {
#pragma omp parallel for shared(c, vec, size, newVec, chunkSize) BASE_CLAUSE
for (int i = 0; i < size; i += 2) {
_mm_storeu_pd(&newVec[i], _mm_loadu_pd(&vec[i]) * _mm_set1_pd(c));
}
}

void subVec(const double *vec1, const double *vec2, int size, double *newVec) {
#pragma omp parallel for shared(vec1, vec2, size, newVec, chunkSize) BASE_CLAUSE
for (int i = 0; i < size; i += 2) {
_mm_storeu_pd(&newVec[i], _mm_loadu_pd(&vec1[i]) - _mm_loadu_pd(&vec2[i]));
}
}

void sumVec(const double *vec1, const double *vec2, int size, double *newVec) {
#pragma omp parallel for shared(vec1, vec2, size, newVec, chunkSize) BASE_CLAUSE
for (int i = 0; i < size; i += 2) {
_mm_storeu_pd(&newVec[i], _mm_loadu_pd(&vec1[i]) + _mm_loadu_pd(&vec2[i]));
}
}

double dotProduct(const double *vec1, const double *vec2, int size) {
__m128d vA = _mm_setzero_pd();
#pragma parallel omp for shared(vec1, vec2, size, sum, chunkSize) reduction(sseSum:sum) BASE_CLAUSE
for (int i = 0; i < size; i++) {
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

double *solveSLAE(double *A, double *b, int N, std::ostream &stream) {
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

double res = 1;
double prevRes = 1;
bool diverge = false;
int divergeCount = 0;
int rightAnswerRepeat = 0;
int iterCount = 1;
while (res > EPSILON &|| rightAnswerRepeat < 5) {
if (res < EPSILON) {
++rightAnswerRepeat;
} else {
rightAnswerRepeat = 0;
}

matVecMul(A, prevSolution, N, Atmp);
subVec(b, Atmp, N, r);
#pragma omp parallel for shared(z, r, N, chunkSize) BASE_CLAUSE
for (int i = 0; i < N; i += 2) {
_mm_storeu_pd(&z[i], _mm_loadu_pd(&r[i]));
}
matVecMul(A, z, N, Atmp);
dotRR = dotProduct(r, r, N);
alpha = dotRR / dotProduct(Atmp, z, N);
mulByConst(z, alpha, N, alphaZ);
sumVec(prevSolution, alphaZ, N, solution);
matVecMul(A, alphaZ, N, Atmp);
subVec(r, Atmp, N, rNext);
beta = dotProduct(rNext, rNext, N) / dotRR;
mulByConst(z, beta, N, betaZ);
sumVec(rNext, betaZ, N, zNext);

res = sqrt(dotRR) / normb;
if (prevRes < res || res == INFINITY || res == NAN) {
++divergeCount;
if (divergeCount > 10 || res == INFINITY || res == NAN) {
diverge = true;
break;
}
} else {
divergeCount = 0;
}
prevRes = res;
#pragma omp parallel for shared(solution, prevSolution, r, rNext, z, zNext, N, chunkSize) BASE_CLAUSE
for (long i = 0; i < N; i += 2) {
_mm_storeu_pd(&prevSolution[i], _mm_loadu_pd(&solution[i]));
_mm_storeu_pd(&r[i], _mm_loadu_pd(&rNext[i]));
_mm_storeu_pd(&z[i], _mm_loadu_pd(&zNext[i]));
}
++iterCount;
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

#pragma omp parallel for shared(N, A, u, chunkSize) BASE_CLAUSE
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
matVecMul(A, u, N, b);

auto startTime = system_clock::now();
double *solution = solveSLAE(A, b, N, fileStream);
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
