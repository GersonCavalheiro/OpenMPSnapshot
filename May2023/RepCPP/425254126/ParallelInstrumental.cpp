#include "algo/interfaces/parallel/ParallelInstrumental.h"

#include <ctime>


bool ParallelInstrumental::isPrime(int num) {
bool ok = true;
for (int i = 2; i <= num / 2; ++i) {
if (num % i == 0) {
ok = false;
break;
}
}

return ok;
}

vec ParallelInstrumental::findDivisors(int num) {
vec res;

for (int i = 2; i <= sqrt(num); i++) {
if (num % i == 0) {
if (num / i != i) {
res.push_back((double)num / i);
}

res.push_back(i);
}
}

return res;
}

void ParallelInstrumental::prepareData() {
size_t n, threadNums;

do {
std::cout << "Enter the dimension (N) = ";
std::cin >> n;

std::cout << "Enter the number of compute nodes (threadNum) = ";
std::cin >> threadNums;
} while(!checkData());

this->prepareData(n, threadNums);
}

void ParallelInstrumental::prepareData(size_t n, size_t tN) {
N = n;
threadNum = tN;
blockSize = (int)N / threadNum;
interSize = (threadNum - 1) * 2;

this->setParallelOptions();
}

void ParallelInstrumental::setParallelOptions() const {
omp_set_dynamic(0);
omp_set_num_threads((int)threadNum);
}

bool ParallelInstrumental::checkData() const {
if (N < 7) {
std::cout << "Dimension (N = " << N << ") is too small\n"
<< "Parallel computing is not effective for it\n"
<< "Enter a larger dimension (N)\n\n";

return false;
}

if ((N / threadNum) < 3) {
std::cout << "The algorithm is not working for the proportions of the dimension (N) "
<< "with the number of computing nodes (threadNum)\n"
<< "Enter a larger dimension (N), or reduce the number of computing nodes (threadNum)\n\n";

return false;
}

vec div = findDivisors((int)N);
if (isPrime((int)N) || std::find(div.begin(), div.end(), threadNum) == div.end()) {
std::cout << "It is impossible to split the dimension (N = "<< N << ") into the same blocks (blockSize)\n"
<< "Enter the correct values of dimension (N) and number of computing nodes (threadNum)\n\n";

return false;
}

return true;
}

vec ParallelInstrumental::createVecN() {
vec a(N);
std::iota(a.begin(), a.end(), 0);

return a;
}

vec ParallelInstrumental::createVecRand() {
std::random_device dev;
std::mt19937 gen(dev());
vec a(N);

#pragma omp parallel for shared(a, N, gen) default(none) if (N > 500)
for (int i = 0; i < N; i++)
a[i] = gen() % 100;

return a;
}

matr ParallelInstrumental::createThirdDiagMatrI() {
matr a(N, vec(N));

#pragma omp parallel for shared(a, N) default(none) if (N > 500)
for (int i = 1; i < N; i++) {
for (int j = 0; j < N; j++) {
a[i][i] = 3.;
a[i][i - 1] = 1.;
a[i - 1][i] = 2.;
}
}

a[0][0] = 1.; a[N - 1][N - 1] = 1.;
a[0][1] = -0.5; a[N - 1][N - 2] = -0.5;

return a;
}

matr ParallelInstrumental::createNewMatr(vec a_, vec c_, vec b_, pairs kappa_, pairs gamma_) {
matr a(N, vec(N));

for (int i = 1; i < N; i++) {
for (int j = 0; j < a_.size(); j++) {
a[i][i] = c_[j];
a[i][i - 1] = a_[j];
a[i - 1][i] = b_[j];
}
}

a[0][0] = gamma_.first; a[N - 1][N - 1] = gamma_.second;
a[0][1] = kappa_.first; a[N - 1][N - 2] = kappa_.second;

return a;
}

vec ParallelInstrumental::createNewVec(vec phi_, pairs mu) {
vec res(N, 0.);

res[0] = mu.first; res[N - 1] = mu.second;
for (int i = 1; i < N - 1; i++) {
res[i] = -phi_[i - 1];
}

return res;
}

matr ParallelInstrumental::createThirdDiagMatrRand() {
std::random_device dev;
std::mt19937 gen(dev());
matr a(N, vec(N));

a[0][0] = gen() % 100;
#pragma omp parallel for shared(a, N, gen) default(none) if (N > 500)
for (int i = 1; i < N; i++) {
for (int j = 0; j < N; j++) {
a[i][i] = gen() % 100;
a[i][i - 1] = gen() % 100;
a[i - 1][i] = gen() % 100;
}
}

return a;
}