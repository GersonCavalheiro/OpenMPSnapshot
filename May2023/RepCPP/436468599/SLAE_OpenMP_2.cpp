#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>

#define TYPE guided
#define CHUNK chunkSize
int matrixSize = 1;
int threadCount = 1;
int chunkSize = 1;

using namespace std::chrono;

void matVecMul(const double *mat, const double *vec, int size, double *newVec) {
for (int i = 0; i < size; i++) {
newVec[i] = 0;
for (int j = 0; j < size; j++) {
newVec[i] += mat[i * size + j] * vec[j];
}
}
}

double dotProduct(const double *vec1, const double *vec2, int size) {
double sum = 0;
for (int i = 0; i < size; ++i) {
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

double *solveSLAE(const double *A, double *b, int N) {
auto *solution = new double[N];
std::fill(solution, solution + N, 0);
auto *prevSolution = new double[N];
std::fill(prevSolution, prevSolution + N, 0);

auto *r = new double[N];
auto *z = new double[N];
auto *rNext = new double[N];
auto *zNext = new double[N];

double alpha;
double beta;

const double EPSILON = 1e-007;

double normb = sqrt(dotProduct(b, b, N));
double dotRR;
double sum = 0;

double res = 1;
double prevRes = 1;
bool diverge = false;
int divergeCount = 0;
int rightAnswerRepeat = 0;
int iterCount = 1;
#pragma omp parallel num_threads(threadCount) firstprivate(rightAnswerRepeat, divergeCount, diverge, res, prevRes)
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
for (int i = 0; i < N; ++i) {
double tmp = 0;
for (int j = 0; j < N; j++) {
tmp += A[i * N + j] * prevSolution[j];
}
r[i] = b[i] - tmp;
z[i] = r[i];
}
#pragma omp single
{
dotRR = 0;
}
#pragma omp for schedule(TYPE, CHUNK) reduction(+:dotRR) reduction(+:sum)
for (int i = 0; i < N; ++i) {
double tmp = 0;
for (int j = 0; j < N; ++j) {
tmp += A[i * N + j] * z[j];
}
dotRR += r[i] * r[i];
sum += tmp * z[i];
}
#pragma omp single
{
alpha = dotRR / sum;
sum = 0;
}
#pragma omp for schedule(TYPE, CHUNK)
for (int j = 0; j < N; j++) {
solution[j] = prevSolution[j] + alpha * z[j];
}
#pragma omp for schedule(TYPE, CHUNK) reduction(+:sum)
for (int j = 0; j < N; j++) {
rNext[j] = r[j] - alpha * A[j * N + j] * z[j];
sum += rNext[j] * rNext[j];
}
#pragma omp single
{
beta = sum / dotRR;
}
#pragma omp for schedule(TYPE, CHUNK)
for (int j = 0; j < N; j++) {
zNext[j] = rNext[j] + beta * z[j];
prevSolution[j] = solution[j];
r[j] = rNext[j];
z[j] = zNext[j];
}

res = sqrt(sum) / normb;
if (prevRes < res || res == INFINITY || res == NAN) {
++divergeCount;
if (divergeCount > 10 || res == INFINITY || res == NAN) {
diverge = true;
break;
}
} else {
divergeCount = 0;
}
#pragma omp single
{
prevRes = res;
++iterCount;
}
}
delete[](prevSolution);
delete[](r);
delete[](z);
delete[](rNext);
delete[](zNext);

std::cout << "iterCount: " << iterCount << std::endl;

if (diverge) {
std::cout << "Does not converge" << std::endl;
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
matVecMul(A, u, N, b);

auto startTime = system_clock::now();
double *prevSolution = solveSLAE(A, b, N);
auto endTime = system_clock::now();
auto duration = duration_cast<nanoseconds>(endTime - startTime);

if (prevSolution != nullptr) {
fileStream << "Answer:" << std::endl;
printMat(u, 1, N, fileStream);
fileStream << "SLAE prevSolution:" << std::endl;
printMat(prevSolution, 1, N, fileStream);
fileStream << "Time: " << duration.count() / double(1000000000) << "sec" << std::endl;
std::cout << "Time: " << duration.count() / double(1000000000) << "sec" << std::endl;
} else {
fileStream << "Does not converge" << std::endl;
}

delete[](prevSolution);
delete[](b);
delete[](u);
delete[](A);
return 0;
}
