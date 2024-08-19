#pragma once

#include <algo/interfaces/parallel/ParallelSweepMethod.h>
#include <test/common/Profiler.h>
#include <test/common/TestRunner.h>
#include <test/common/BaseComponentTest.h>
#include "SerialAlgorithmComponentTest.h"

class ParallelAlgorithmComponentTest final : public ParallelSweepMethod, public BaseComponentTest {
private:
void prepareParallelDataForTest(const ParallelSweepMethod& sweepMethod) {
std::tie(N,
threadNum, blockSize, interSize,
A, b, y) = sweepMethod.getAllFields();
}

void setParallelFields(ParallelSweepMethod& sweepMethod) {
sweepMethod.setAllFields(N, threadNum, blockSize, interSize, A, b, y);
}

public:
ParallelAlgorithmComponentTest() : ParallelSweepMethod() {}

void testTransformation(int n, int tN) {
matr A1, A2;
vec b1, b2;

{
LOG_DURATION("serial")

ParallelSweepMethod psm(n, tN);
this->prepareParallelDataForTest(psm);

for (size_t k = 0; k < n; k += blockSize) {
size_t StartInd = k;
size_t EndInd = k + blockSize;

for (size_t j = StartInd; j < EndInd - 1; j++) {
for (size_t i = j + 1; i < EndInd; i++) {
double temp = A[i][j] / A[j][j];
for (size_t o = 0; o < n; o++) {
A[i][o] -= temp * A[j][o];
}
b[i] -= temp * b[j];
}
}

for (size_t j = EndInd - 1; j > StartInd; j--) {
for (size_t ii = j; ii > StartInd; ii--) {
size_t i = ii - 1;
double temp = A[i][j] / A[j][j];
for (size_t o = 0; o < n; o++) {
A[i][o] -= temp * A[j][o];
}
b[i] -= temp * b[j];
}
}
}

A1 = A;
b1 = b;
}

{
LOG_DURATION("parallel")

ParallelSweepMethod psm(n, tN);
this->prepareParallelDataForTest(psm);

size_t i, j, k, h;
double coef;
size_t iter, start, end;

#pragma omp parallel private(i, j, k, h, iter, start, end, coef) shared(blockSize, N) default(none)
{
iter  = omp_get_thread_num() * blockSize;
start = iter;
end   = iter + blockSize;

for (j = start; j < end - 1; j++) {
for (i = j + 1; i < end; i++) {
coef = A[i][j] / A[j][j];
for (k = 0; k < N; k++) {
A[i][k] -= coef * A[j][k];
}
b[i] -= coef * b[j];
}
}

for (j = end - 1; j > start; j--) {
for (h = j; h > start; h--) {
i = h - 1;
coef = A[i][j] / A[j][j];
for (k = 0; k < N; k++) {
A[i][k] -= coef * A[j][k];
}
b[i] -= coef * b[j];
}
}
}

A2 = A;
b2 = b;
}

Instrumental::compareMatr(A1, A2);
Instrumental::compareVec(b1, b2);
}

void testTransformationByTask7() {
SerialInstrumental si(5);
ParallelSweepMethod psm(6, 1);
this->prepareParallelDataForTest(psm);

A = {
{1.000,    0.000,    0.000,    0.000,    0.000,    0.000},
{300.000, -605.000,  300.000,  0.000,    0.000,    0.000},
{0.000,    300.000, -605.000,  300.000,  0.000,    0.000},
{0.000,    0.000,    300.000, -605.000,  300.000,  0.000},
{0.000,    0.000,    0.000,    300.000, -605.000,  300.000},
{0.000,    0.000,    0.000,    0.000,    0.000,    1.000}
};
b = {10.0, 2092.0, 2038.0, 1948.0, 1822.0, 100.0};
u = {10, 13.6, 24.4, 42.4, 67.6, 100};
this->setParallelFields(psm);

psm.transformation();

this->prepareParallelDataForTest(psm);

y.resize(A.size());
for (int i = 0; i < A.size(); i++) {
y[i] = b[i] / A[i][i];
}

Instrumental::compareVec(y, u);
}

std::pair<matr, vec> testCollectInterferElemPreprocessing(int n, int tN) {
ParallelSweepMethod psm(n, tN);

psm.transformation();
this->prepareParallelDataForTest(psm);

matr R1, R2;
vec partB1, partB2;

{
LOG_DURATION("serial")

matr R(interSize, vec(interSize, 0.));
vec partB(interSize);

if (tN > 2) {
size_t bS1 = blockSize - 1;
R[0][0] = A[bS1][bS1];
R[0][1] = A[bS1][blockSize];
R[1][0] = A[blockSize][bS1];
R[2][1] = A[2 * blockSize - 1][bS1];

size_t bSN = N - blockSize;
R[interSize - 1][interSize - 1] = A[bSN][bSN];
R[interSize - 1][interSize - 2] = A[bSN][bSN - 1];
R[interSize - 2][interSize - 1] = A[bSN - 1][bSN];
R[interSize - 3][interSize - 2] = A[N - blockSize * 2][bSN];

for (size_t i = blockSize, k = 1; i < N; i += blockSize, k += 2) {
partB[k - 1] = b[i - 1];
partB[k] = b[i];
}

} else {
size_t bS1 = blockSize - 1;

R[0][0] = A[bS1][bS1];
R[0][1] = A[bS1][blockSize];
R[1][0] = A[blockSize][bS1];
R[1][1] = A[blockSize][blockSize];

for (size_t i = blockSize, k = 1; i < N; i += blockSize, k += 2) {
partB[k - 1] = b[i - 1];
partB[k] = b[i];
}
}

R1 = R;
partB1 = partB;
}

{
LOG_DURATION("parallel")

matr R(interSize, vec(interSize, 0.));
vec partB(interSize);

size_t iter;
size_t s;

if (tN > 2) {
#pragma omp parallel private(s, iter) shared(R, partB) num_threads(threadNum - 1) default(none)
{
iter = omp_get_thread_num();

#pragma omp taskgroup
{
#pragma omp task shared(R) default(none)
this->preULR(R);

#pragma omp task shared(R) default(none)
this->preLRR(R);

#pragma omp taskloop private(s) firstprivate(iter) shared(partB) default(none)
for (s = iter; s < iter + 1; s++) {
partB[2 * s] = b[(s + 1) * blockSize - 1];
partB[2 * s + 1] = b[(s + 1) * blockSize];
}
}
}
} else {
size_t bS1 = blockSize - 1;

R[0][0] = A[bS1][bS1];
R[0][1] = A[bS1][blockSize];
R[1][0] = A[blockSize][bS1];
R[1][1] = A[blockSize][blockSize];

size_t i, k = 0;

#pragma omp parallel for private(i, iter) firstprivate(k) shared(N, blockSize, b, partB) default(none)
for (i = (omp_get_thread_num() + 1) * blockSize; i < N; i += iter) {
partB[k++] = b[i - 1];
partB[k++] = b[i];
}
}

R2 = R;
partB2 = partB;
}

Instrumental::compareMatr(R1, R2);
Instrumental::compareVec(partB1, partB2);

return std::make_pair(R1, partB1);
}

std::pair<matr, vec> testCollectInterferElemPostprocessing(int n, int tN) {
ParallelSweepMethod psm(n, tN);
psm.transformation();

matr R, R1, R2;
vec partB;

{
LOG_DURATION("serial")

std::tie(R, partB) = this->testCollectInterferElemPreprocessing(n, tN);
this->prepareParallelDataForTest(psm);

if (tN > 2) {
for (size_t i = blockSize, k = 1; i < N - blockSize; i += blockSize, k += 2) {
for (size_t j = blockSize, l = 1; j < N - blockSize; j += blockSize, l += 2) {

double a1 = A[i][j];
double a2 = A[i][j + blockSize - 1];
double a3 = A[i + blockSize - 1][j];
double a4 = A[i + blockSize - 1][j + blockSize - 1];

if (a1 != 0 && a4 != 0) {
R[k][l] = a1;
R[k + 1][l + 1] = a4;
} else if (a1 != 0) {
R[k][l - 1] = a1;
R[k + 1][l] = a3;
} else if (a4 != 0) {
R[k][l + 1] = a2;
R[k + 1][l + 2] = a4;
}
}
}
}

R1 = R;
}

{
LOG_DURATION("parallel")

std::tie(R, partB) = this->testCollectInterferElemPreprocessing(n, tN);
this->prepareParallelDataForTest(psm);

size_t k = 1, l = 1;
size_t iter;
size_t i, j;
double a1, a2, a3, a4;

if (threadNum > 2) {
#pragma omp parallel private(iter, i, j, a1, a2, a3, a4) firstprivate(k, l) shared(A, R, blockSize) num_threads(threadNum - 2) default(none)
{
iter = (omp_get_thread_num() + 1) * blockSize;

for (i = iter; i < N - blockSize; i += iter) {
for (j = iter; j < N - blockSize; j += iter) {
a1 = A[i][j];
a2 = A[i][j + blockSize - 1];
a3 = A[i + blockSize - 1][j];
a4 = A[i + blockSize - 1][j + blockSize - 1];

if (a1 != 0 && a4 != 0) {
R[k][l] = a1;
R[k + 1][l + 1] = a4;
} else if (a1 != 0) {
R[k][l - 1] = a1;
R[k + 1][l] = a3;
} else if (a4 != 0) {
R[k][l + 1] = a2;
R[k + 1][l + 2] = a4;
}

l += 2;
}

l = 1;
k += 2;
}
}
}

R2 = R;
}

Instrumental::compareMatr(R1, R2);

return std::make_pair(R, partB);
}

void testOrderingCoefficient(int n, int tN) {
ParallelSweepMethod psm(n, tN);
matr R, R1, R2;
vec partB;

psm.transformation();

{
LOG_DURATION("serial")

std::tie(R, partB) = this->testCollectInterferElemPostprocessing(n, tN);
this->prepareParallelDataForTest(psm);

for (int i = 0; i < interSize; i += 2) {
std::swap(R[i][i], R[i][i + 1]);
std::swap(R[i + 1][i], R[i + 1][i + 1]);
}

R1 = R;
}

{
LOG_DURATION("parallel")

std::tie(R, partB) = this->testCollectInterferElemPostprocessing(n, tN);
this->prepareParallelDataForTest(psm);

#pragma omp parallel for shared(R, partB) default(none)
for (int i = 0; i < interSize; i += 2) {
std::swap(R[i][i], R[i][i + 1]);
std::swap(R[i + 1][i], R[i + 1][i + 1]);
}

R2 = R;
}

Instrumental::compareMatr(R1, R2);
}

static vec testCollectPartY() {
matr R = {
{1.000,    0.000,    0.000,    0.000,    0.000,    0.000},
{300.000, -605.000,  300.000,  0.000,    0.000,    0.000},
{0.000,    300.000, -605.000,  300.000,  0.000,    0.000},
{0.000,    0.000,    300.000, -605.000,  300.000,  0.000},
{0.000,    0.000,    0.000,    300.000, -605.000,  300.000},
{0.000,    0.000,    0.000,    0.000,    0.000,    1.000}
};
vec partB = {10.0, 2092.0, 2038.0, 1948.0, 1822.0, 100.0};
vec x = {10.0, 13.6, 24.4, 42.4, 67.6, 100.0};
size_t interfereSize = 6;

vec y1, y2;

{
LOG_DURATION("serial")

vec a(interfereSize - 2),
c(interfereSize - 2),
b(interfereSize - 2),
phi(interfereSize - 2);

pairs mu    = std::make_pair(partB[0], partB[interfereSize - 1]);
pairs kappa = std::make_pair(-R[0][1], -R[interfereSize - 1][interfereSize - 2]);
pairs gamma = std::make_pair(R[0][0], R[interfereSize - 1][interfereSize - 1]);

for (size_t i = 1; i < interfereSize - 1; i++) {
a[i - 1] = R[i][i - 1];
c[i - 1] = -R[i][i];
b[i - 1] = R[i][i + 1];
phi[i - 1] = -partB[i];
}

SerialSweepMethod ssm(a, c, b, phi, kappa, mu, gamma);
vec res = ssm.run();

for (int i = 0; i < interfereSize - 1; i += 2) {
std::swap(res[i], res[i + 1]);
}

y1 = res;
}

{
LOG_DURATION("parallel")

size_t i;
vec a(interfereSize - 2),
c(interfereSize - 2),
b(interfereSize - 2),
phi(interfereSize - 2);

pairs mu    = std::make_pair(partB[0], partB[interfereSize - 1]);
pairs kappa = std::make_pair(-R[0][1], -R[interfereSize - 1][interfereSize - 2]);
pairs gamma = std::make_pair(R[0][0], R[interfereSize - 1][interfereSize - 1]);

#pragma omp parallel for private(i) shared(a, c, b, phi, partB, R, interfereSize) num_threads(4) default(none)
for (i = 1; i < interfereSize - 1; i++) {
a[i - 1] = R[i][i - 1];
b[i - 1] = R[i][i + 1];
c[i - 1] = -R[i][i];
phi[i - 1] = -partB[i];
}

SerialSweepMethod ssm(a, c, b, phi, kappa, mu, gamma);
vec res = ssm.run();

#pragma omp parallel for private(i) shared(y1, interfereSize, res) num_threads(4) default(none)
for (i = 0; i < interfereSize - 1; i += 2) {
std::swap(res[i], res[i + 1]);
}

y2 = res;
}

Instrumental::compareVec(y1, y2);

return y1;
}

void testCollectNotInterferElemPreprocessing(int n, int tN) {
vec y1, y2;

{
ParallelSweepMethod psm(n, tN);
psm.transformation();
this->prepareParallelDataForTest(psm);

LOG_DURATION("serial");

size_t last = N - blockSize - 1;
for (size_t i = 0; i < blockSize - 1; i++) {
size_t j = N - i - 1;

b[i] -= A[i][blockSize] * y[blockSize];
b[j] -= A[j][last] * y[last];

y[i] = b[i] / A[i][i];
y[j] = b[j] / A[j][j];
}

y1 = y;
}

{
ParallelSweepMethod psm(n, tN);
psm.transformation();
this->prepareParallelDataForTest(psm);

LOG_DURATION("parallel")

size_t i, j;
size_t last = N - blockSize - 1;

#pragma omp parallel for private(i, j) firstprivate(last) shared(blockSize, N, b, A, y) num_threads(2) default(none)
for (i = 0; i < blockSize - 1; i++) {
j = N - i - 1;

b[i] -= A[i][blockSize] * y[blockSize];
b[j] -= A[j][last] * y[last];

y[i] = b[i] / A[i][i];
y[j] = b[j] / A[j][j];
}

y2 = y;
}

Instrumental::compareVec(y1, y2);
}

void testCollectNotInterferElemPostprocessing(int n, int tN) {
vec y1, y2;

{
ParallelSweepMethod psm(n, tN);
psm.transformation();
this->prepareParallelDataForTest(psm);

size_t i, j;

LOG_DURATION("serial")

for (i = blockSize + 1; i < N - blockSize; i += blockSize) {
for (j = i; j < i + blockSize - 2; j++) {
b[j] -= (A[j][i - 2] * y[i - 2] + A[j][i + blockSize - 1] * y[i + blockSize - 1]);

y[j] = b[j] / A[j][j];
}
}

y1 = y;
}

{
ParallelSweepMethod psm(n, tN);
psm.transformation();
this->prepareParallelDataForTest(psm);

size_t i, j;

LOG_DURATION("parallel")

if (threadNum > 2) {
#pragma omp parallel private(i, j) shared(blockSize, N, b, A, y) num_threads(threadNum - 2) default(none)
{
i = (omp_get_thread_num() + 1) * blockSize + 1;

for (j = i; j < i + blockSize - 2; j++) {
b[j] -= (A[j][i - 2] * y[i - 2] + A[j][i + blockSize - 1] * y[i + blockSize - 1]);

y[j] = b[j] / A[j][j];
}
}
}

y2 = y;
}

Instrumental::compareVec(y1, y2);
}

void testCollectNotInterferElem(int n, int tN) {
ParallelSweepMethod psm(n, tN);
psm.transformation();
this->prepareParallelDataForTest(psm);

size_t i, j;
size_t last = N - blockSize - 1;

#pragma omp parallel for private(i, j) firstprivate(last) shared(blockSize, N, b, A, y) num_threads(2) default(none)
for (i = 0; i < blockSize - 1; i++) {
j = N - i - 1;

b[i] -= A[i][blockSize] * y[blockSize];
b[j] -= A[j][last] * y[last];

y[i] = b[i] / A[i][i];
y[j] = b[j] / A[j][j];
}

if(threadNum > 2) {
#pragma omp parallel private(i, j) shared(blockSize, N, b, A, y) num_threads(threadNum - 2) default(none)
{
i = (omp_get_thread_num() + 1) * blockSize + 1;

for (j = i; j < i + blockSize - 2; j++) {
b[j] -= (A[j][i - 2] * y[i - 2] + A[j][i + blockSize - 1] * y[i + blockSize - 1]);

y[j] = b[j] / A[j][j];
}
}
}
}

void testCollectFullY(int n, int tN) {
vec y1, y2;

{
LOG_DURATION("serial")

this->testCollectNotInterferElem(n, tN);
vec partY((tN - 1) * 2, 0);
std::iota(partY.begin(), partY.end(), 0);

size_t iter, k = 0;

for (iter = 0; iter < tN - 1; iter++) {
size_t i = (iter + 1) * blockSize;
y[i - 1] = partY[k++];
y[i] = partY[k++];
}

y1 = y;
}

{
LOG_DURATION("parallel")

this->testCollectNotInterferElem(n, tN);
vec partY((tN - 1) * 2, 1.);
std::iota(partY.begin(), partY.end(), 0);

size_t i, iter;
size_t k = 2;

#pragma omp parallel private(i, iter) firstprivate(k) shared(blockSize, y, partY) num_threads(threadNum - 1) default(none)
{
iter = omp_get_thread_num();
i = (iter + 1) * blockSize;

y[i - 1] = partY[iter * k];
y[i] = partY[iter * k + 1];
}

y2 = y;
}

Instrumental::compareVec(y1, y2);
}

void testTimeSerial(int n) {
SerialAlgorithmComponentTest s;

{
LOG_DURATION("serial = " + std::to_string(n) + ": ")
s.testTask7(n);
}

}

void testTimeParallel(int n, int tN) {
SerialAlgorithmComponentTest s;

vec A, C, B, phi1, v, Phi;

pairs mu = std::make_pair(10., 100.);
pairs kappa = std::make_pair(0., 0.);
pairs gamma = std::make_pair(1., 1.);

std::tie(A, C, B, phi1, v, Phi) = s.testTask7(n);

for (double & i : C) {
i = -i;
}

ParallelSweepMethod psm(n, A, C, B, phi1, kappa, mu, gamma, tN);

{
LOG_DURATION(std::to_string(n) + ", " + to_string(tN) + ": ")
psm.run();
}
}

void testFullAlgorithm(int n, int tN) {
ParallelSweepMethod psm(n, tN);
this->prepareParallelDataForTest(psm);

A = {
{1.000,    0.000,    0.000,    0.000,    0.000,    0.000},
{1.000, -2.000,  3.000,  0.000,    0.000,    0.000},
{0.000,    1, -2,  3.000,  0.000,    0.000},
{0.000,    0.000,    1.000, -2.000,  3.000,  0.000},
{0.000,    0.000,    0.000,    1.000, -2.000,  3.000},
{0.000,    0.000,    0.000,    0.000,    0.000,    1.000}
};
b = {10.0, 1.0, 2.0, 3.0, 4.0, 100.0};

this->setParallelFields(psm);
y = psm.run();
printVec(y, "test full algorithm");
}


void testTask7(int n, int tN) {
ParallelSweepMethod psm(n, tN);
this->prepareParallelDataForTest(psm);

A = {
{1.000,        0,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000},
{300.000, -605.0,  300.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000},
{0.000,  300.000, -605.000,  300.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000},
{0.000,    0.000,  300.000, -605.000,  300.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000},
{0.000,    0.000,    0.000,  300.000, -605.000,  300.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000},
{0.000,    0.000,    0.000,    0.000,  300.000, -605.000,  300.000,    0.000,    0.000,    0.000,    0.000,    0.000},
{0.000,    0.000,    0.000,    0.000,    0.000,  300.000, -605.000,  300.000,    0.000,    0.000,    0.000,    0.000},
{0.000,    0.000,    0.000,    0.000,    0.000,    0.000,  300.000, -605.000,  300.000,    0.000,    0.000,    0.000},
{0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,  300.000, -605.000,  300.000,    0.000,    0.000},
{0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,  300.000, -605.000,  300.000,    0.000},
{0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,  300.000, -605.000,  300.000},
{0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,    0.000,        0,    1.000}
};
b = {10.0, 2106.28, 2095.12, 2076.53, 2050.5, 2017.02, 1976.12, 1927.77, 1871.98, 1808.76, 1738.1, 100.0};
vec res = {10, -13.465, -30.133, -40.32, -44.258, -42.098, -33.916, -19.712, 0.588, 27.139, 60.171, 100};

this->setParallelFields(psm);


for (size_t k = 0; k < N; k += blockSize) {
size_t StartInd = k;
size_t EndInd = k + blockSize;

for (size_t j = StartInd; j < EndInd - 1; j++) {
for (size_t i = j + 1; i < EndInd; i++) {
double temp = A[i][j] / A[j][j];
for (size_t o = 0; o < N; o++) {
A[i][o] -= temp * A[j][o];
}
b[i] -= temp * b[j];
}
}

for (size_t j = EndInd - 1; j > StartInd; j--) {
for (size_t ii = j; ii > StartInd; ii--) {
size_t i = ii - 1;
double temp = A[i][j] / A[j][j];
for (size_t o = 0; o < N; o++) {
A[i][o] -= temp * A[j][o];
}
b[i] -= temp * b[j];
}
}
}


matr R(interSize, vec(interSize, 0.));
vec partB(interSize);

if (threadNum > 2) {
size_t bS1 = blockSize - 1;
R[0][0] = A[bS1][bS1];
R[0][1] = A[bS1][blockSize];
R[1][0] = A[blockSize][bS1];
R[2][1] = A[2 * blockSize - 1][bS1];

size_t bSN = N - blockSize;
R[interSize - 1][interSize - 1] = A[bSN][bSN];
R[interSize - 1][interSize - 2] = A[bSN][bSN - 1];
R[interSize - 2][interSize - 1] = A[bSN - 1][bSN];
R[interSize - 3][interSize - 2] = A[N - blockSize * 2][bSN];

for (size_t i = blockSize, k = 1; i < N; i += blockSize, k += 2) {
partB[k - 1] = b[i - 1];
partB[k] = b[i];
}

for (size_t i = blockSize, k = 1; i < N - blockSize; i += blockSize, k += 2) {
for (size_t j = blockSize, l = 1; j < N - blockSize; j += blockSize, l += 2) {

double a1 = A[i][j];
double a2 = A[i][j + blockSize - 1];
double a3 = A[i + blockSize - 1][j];
double a4 = A[i + blockSize - 1][j + blockSize - 1];

if (i == j) {
R[k][l] = a1;
R[k + 1][l + 1] = a4;
} else if (i < j) {
R[k][l - 1] = a1;
R[k + 1][l] = a3;
} else {
R[k][l + 1] = a2;
R[k + 1][l + 2] = a4;
}
}
}

} else {
size_t bS1 = blockSize - 1;

R[0][0] = A[bS1][bS1];
R[0][1] = A[bS1][blockSize];
R[1][0] = A[blockSize][bS1];
R[1][1] = A[blockSize][blockSize];

for (size_t i = blockSize, k = 1; i < N; i += blockSize, k += 2) {
partB[k - 1] = b[i - 1];
partB[k] = b[i];
}
}

for (int i = 0; i < interSize; i += 2) {
std::swap(R[i][i], R[i][i + 1]);
std::swap(R[i + 1][i], R[i + 1][i + 1]);
}


vec a(interSize - 2),
c(interSize - 2),
b_(interSize - 2),
phi(interSize - 2);

pairs mu    = std::make_pair(partB[0], partB[interSize - 1]);
pairs kappa = std::make_pair(-R[0][1], -R[interSize - 1][interSize - 2]);
pairs gamma = std::make_pair(R[0][0], R[interSize - 1][interSize - 1]);

for (size_t i = 1; i < interSize - 1; i++) {
a[i - 1] = R[i][i - 1];
c[i - 1] = -R[i][i];
b_[i - 1] = R[i][i + 1];
phi[i - 1] = -partB[i];
}

SerialSweepMethod ssm(a, c, b_, phi, kappa, mu, gamma);
vec partY = ssm.run();

for (int i = 0; i < interSize - 1; i += 2) {
std::swap(partY[i], partY[i + 1]);
}

size_t i, j;
size_t iter, k = 0;

for (iter = 0; iter < tN - 1; iter++) {
i = (iter + 1) * blockSize;
y[i - 1] = partY[k++];
y[i] = partY[k++];
}

size_t last = N - blockSize - 1;
for (i = 0; i < blockSize - 1; i++) {
j = N - i - 1;

b[i] -= A[i][blockSize] * y[blockSize];
b[j] -= A[j][last] * y[last];

y[i] = b[i] / A[i][i];
y[j] = b[j] / A[j][j];
}

for (i = blockSize + 1; i < N - blockSize; i += blockSize) {
for (j = i; j < i + blockSize - 2; j++) {
b[j] -= (A[j][i - 2] * y[i - 2] + A[j][i + blockSize - 1] * y[i + blockSize - 1]);

y[j] = b[j] / A[j][j];
}
}


k = 0;
for (iter = 0; iter < tN - 1; iter++) {
i = (iter + 1) * blockSize;
y[i - 1] = partY[k++];
y[i] = partY[k++];
}

Instrumental::compareVec(y, res);
}

void execute() {
std::vector<std::function<void()>> tests = {
[this]() { this->testFullAlgorithm(6, 2); }
};

BaseComponentTest::execute(tests, "Parallel Component Tests");
}

void executeTime() {

std::vector<int> nums = { 840, 1680, 3360, 5040 };

for (int i = 0; i < 2; i++ ) {
std::vector<std::function<void()>> tests = {
[this, &nums]() { this->testTimeParallel(nums[3], 8); }
};

BaseComponentTest::execute(tests, "Tests");
}
}
};