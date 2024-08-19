#include <test/common/Profiler.h>
#include "algo/interfaces/parallel/ParallelSweepMethod.h"
#include "algorithm"

std::tuple<size_t, size_t, size_t, size_t, matr, vec, vec> ParallelSweepMethod::getAllFields() const {
return std::make_tuple(N, threadNum, blockSize, interSize, A, b, y);
}

void ParallelSweepMethod::setAllFields(size_t N, size_t threadNum, size_t blockSize, size_t interSize, const matr& A_, const vec& b_, const vec& y_) {
this->N = N;
this->threadNum = threadNum;
this->blockSize = blockSize;
this->interSize = interSize;
this->A = A_;
this->b = b_;
this->y = y_;
}

void ParallelSweepMethod::transformation() {
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

}

void ParallelSweepMethod::preULR(matr& R) {
size_t bS1 = blockSize - 1;
R[0][0] = A[bS1][bS1];
R[0][1] = A[bS1][blockSize];
R[1][0] = A[blockSize][bS1];
R[2][1] = A[2 * blockSize - 1][bS1];
}

void ParallelSweepMethod::preLRR(matr& R) {
size_t bSN = N - blockSize;
R[interSize - 1][interSize - 1] = A[bSN][bSN];
R[interSize - 1][interSize - 2] = A[bSN][bSN - 1];
R[interSize - 2][interSize - 1] = A[bSN - 1][bSN];
R[interSize - 3][interSize - 2] = A[N - blockSize * 2][bSN];
}

std::pair<matr, vec> ParallelSweepMethod::collectInterferElem() {
matr R(interSize, vec(interSize, 0.));
vec partB(interSize);

size_t k = 1, l = 1;
size_t iter;
size_t i, j, s;
double a1, a2, a3, a4;

if (threadNum > 2) {

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

#pragma omp parallel private(iter, i, j, a1, a2, a3, a4) firstprivate(k, l) shared(A, R, blockSize) default(none)
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
} else {
size_t bS1 = blockSize - 1;

R[0][0] = A[bS1][bS1];
R[0][1] = A[bS1][blockSize];
R[1][0] = A[blockSize][bS1];
R[1][1] = A[blockSize][blockSize];

k = 0;

#pragma omp parallel for private(i, iter) firstprivate(k) shared(N, blockSize, b, partB) default(none)
for (i = (omp_get_thread_num() + 1) * blockSize; i < N; i += iter) {
partB[k++] = b[i - 1];
partB[k++] = b[i];
}
}

#pragma omp parallel for private(i) shared(R, partB, interSize) default(none)
for (i = 0; i < interSize; i += 2) {
std::swap(R[i][i], R[i][i + 1]);
std::swap(R[i + 1][i], R[i + 1][i + 1]);
}

return std::make_pair(R, partB);
}

vec ParallelSweepMethod::collectPartY(const matr& R, const vec& partB) {
vec a_(interSize - 2),
c_(interSize - 2),
b_(interSize - 2),
phi(interSize - 2);

size_t i;
pairs kappa = std::make_pair(-R[0][1], -R[interSize - 1][interSize - 2]);
pairs mu = std::make_pair(partB[0], partB[interSize - 1]);
pairs gamma = std::make_pair(R[0][0], R[interSize - 1][interSize - 1]);

#pragma omp parallel for private(i) shared(a_, c_, b_, phi, partB, R, interSize) default(none)
for (i = 1; i < interSize - 1; i++) {
a_[i - 1] = R[i][i - 1];
b_[i - 1] = R[i][i + 1];
c_[i - 1] = -R[i][i];
phi[i - 1] = -partB[i];
}

SerialSweepMethod ssm(a_, c_, b_, phi, kappa, mu, gamma);
vec partY = ssm.run();

#pragma omp parallel for private(i) shared(partY, interSize) default(none)
for (i = 0; i < interSize - 1; i += 2) {
std::swap(partY[i], partY[i + 1]);
}

return partY;
}

void ParallelSweepMethod::collectNotInterferElem() {
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

vec ParallelSweepMethod::collectFullY(const vec& partY) {
size_t i, iter;
size_t k = 2;

#pragma omp parallel private(i, iter) firstprivate(k) shared(blockSize, y, partY) num_threads(threadNum - 1) default(none)
{
iter = omp_get_thread_num();
i = (iter + 1) * blockSize;

y[i - 1] = partY[iter * k];
y[i] = partY[iter * k + 1];
}

return y;
}

vec ParallelSweepMethod::run() {
this->transformation();

matr R; vec partB;
std::tie(R, partB) = this->collectInterferElem();

vec partY = this->collectPartY(R, partB);

this->collectFullY(partY);
this->collectNotInterferElem();

return this->collectFullY(partY);
}