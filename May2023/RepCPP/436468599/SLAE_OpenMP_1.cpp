#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>

#define TYPE guided
#define CHUNK chunkSize
#define BASE_CLAUSE default(none) num_threads(threadCount) schedule(TYPE, CHUNK)
int matrixSize = 1;
int threadCount = 1;
int chunkSize = 1;

using namespace std::chrono;

void matVecMul(const double *mat, const double *vec, int size, double *newVec) {
#pragma omp parallel for shared(mat, vec, size, newVec, chunkSize) BASE_CLAUSE
for (int i = 0; i < size; i++) {
newVec[i] = 0;
for (int j = 0; j < size; j++) {
newVec[i] += mat[i * size + j] * vec[j];
}
}
}

void mulByConst(double c, const double *vec, long size, double *newVec) {
#pragma omp parallel for shared(c, vec, size, newVec, chunkSize) BASE_CLAUSE
for (int i = 0; i < size; i++) {
newVec[i] = vec[i] * c;
}
}

void sumVec(const double *vec1, const double *vec2, long size, double *newVec) {
#pragma omp parallel for shared(vec1, vec2, size, newVec, chunkSize) BASE_CLAUSE
for (int i = 0; i < size; i++) {
newVec[i] = vec1[i] + vec2[i];
}
}

void subVec(const double *vec1, const double *vec2, long size, double *newVec) {
#pragma omp parallel for shared(vec1, vec2, size, newVec, chunkSize) BASE_CLAUSE
for (int i = 0; i < size; i++) {
newVec[i] = vec1[i] - vec2[i];
}
}

double dotProduct(const double *vec1, const double *vec2, int size) {
double sum = 0;
#pragma parallel omp for shared(vec1, vec2, size, sum, chunkSize) reduction(+:sum) BASE_CLAUSE
for (int i = 0; i < size; i++) {
sum += vec1[i] * vec2[i];
}
return sum;
}

void printMat(double *mat, int rows, int columns, std::ostream &stream) {
for (int i = 0; i < rows; ++i) {
for (int j = 0; j < columns; ++j) {
stream << mat[i * columns + j] << " ";
}
stream << std::endl;
}
}

double *solveSLAE(double *A, double *b, int N) {
auto *nextSolution = new double[N];
std::fill(nextSolution, nextSolution + N, 0);
auto *solution = new double[N];
std::fill(solution, solution + N, 0);

auto *Atmp = new double[N];
auto *r = new double[N];
auto *z = new double[N];
auto *alphaZ = new double[N];
auto *betaZ = new double[N];
auto *rNext = new double[N];
auto *zNext = new double[N];

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
while (res > EPSILON || rightAnswerRepeat < 5) {
if (res < EPSILON) {
++rightAnswerRepeat;
} else {
rightAnswerRepeat = 0;
}

matVecMul(A, solution, N, Atmp);
subVec(b, Atmp, N, r);
#pragma omp parallel for shared(z, r, N, chunkSize) BASE_CLAUSE
for (int i = 0; i < N; ++i) {
z[i] = r[i];
}
matVecMul(A, z, N, Atmp);
dotRR = dotProduct(r, r, N);
alpha = dotRR / dotProduct(Atmp, z, N);
mulByConst(alpha, z, N, alphaZ);
sumVec(solution, alphaZ, N, nextSolution);
matVecMul(A, alphaZ, N, Atmp);
subVec(r, Atmp, N, rNext);
beta = dotProduct(rNext, rNext, N) / dotRR;
mulByConst(beta, z, N, betaZ);
sumVec(rNext, betaZ, N, zNext);

res = sqrt(dotRR) / normb;
prevRes = res;
if (prevRes < res || res == INFINITY || res == NAN) {
++divergeCount;
if (divergeCount > 10 || res == INFINITY || res == NAN) {
diverge = true;
break;
}
} else {
divergeCount = 0;
}
#pragma omp parallel for shared(solution, r, rNext, z, zNext, nextSolution, N, chunkSize) BASE_CLAUSE
for (int i = 0; i < N; i++) {
solution[i] = nextSolution[i];
r[i] = rNext[i];
z[i] = zNext[i];
}
++iterCount;
}
delete[](solution);
delete[](Atmp);
delete[](r);
delete[](z);
delete[](rNext);
delete[](zNext);
delete[](alphaZ);
delete[](betaZ);

std::cout << "iterCount: " << iterCount << std::endl;

if (diverge) {
std::cout << "Does not converge" << std::endl;
delete[](nextSolution);
return nullptr;
} else {
return nextSolution;
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
