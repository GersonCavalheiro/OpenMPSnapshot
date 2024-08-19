#include "kmeans.h"
double getEvDist(const double *x1, const double *x2, const int m) {
double d, r = 0;
int i = 0;
while (i++ < m) {
d = *(x1++) - *(x2++);
r += d * d;
}
return sqrt(r);
}
void elementaryAutoscaling(double *x, const int n, const int m, const int id) {
double Ex = 0, Exx = 0, sd;
int i;
for (i = id; i < n * m; i += m) {
sd = x[i];
Ex += sd;
Exx += sd * sd;
}
Exx /= n;
Ex /= n;
sd = sqrt(Exx - Ex * Ex);
for (i = id; i < n * m; i += m) {
x[i] = (x[i] - Ex) / sd;
}
}
void autoscalingOpenMP(double *x, const int n, const int m) {
int i;
omp_set_nested(0);
#pragma omp parallel for simd shared(x, m) firstprivate(n) private(i)
for (i = 0; i < m; i++) {
elementaryAutoscaling(x, n, m, i);
}
}
int getCluster(const double *x, const double *c, const int m, const int k) {
double curD, minD = DBL_MAX;
int counter, res;
counter = res = 0;
while (counter < k) {
curD = getEvDist(x, c, m);
if (curD < minD) {
minD = curD;
res = counter;
}
counter++;
c += m;
}
return res;
}
void detCoresOpenMP(const double *x, double *c, const int *sn, const int k, const int m) {
int i;
omp_set_nested(0);
#pragma omp parallel for simd shared(c, k) firstprivate(x, sn, m) private(i)
for (i = 0; i < k; i++) {
memcpy(&c[i * m], &x[sn[i] * m], m * sizeof(double));
}
}
void detStartSplittingSimple(const double *x, const double *c, int *y, int *nums, const int m, const int k, const int id) {
int cur;
cur = getCluster(&x[id * m], &c[0], m, k);
y[id] = cur;
nums[cur]++;
}
void detStartSplittingOpenMP(const double *x, const double *c, int *y, int *nums, const int n, const int m, const int k) {
int i;
omp_set_nested(0);
#pragma omp parallel for simd shared(nums, y, n) firstprivate(x, c, m, k) private(i)
for (i = 0; i < n; i++) {
detStartSplittingSimple(x, c, y, nums, m, k, i);
}
}
void simpleCalcCores(const double *x, double *c, const int *res, const int *nums, const int m, const int id) {
int j;
const int buf1 = nums[res[id]], buf2 = res[id] * m, buf3 = id * m;
for (j = 0; j < m; j++) {
c[buf2 + j] += x[buf3 + j] / buf1;
}
}
void calcCoresOpenMP(const double *x, double *c, const int *y, const int *nums, const int n, const int m) {
int i;
omp_set_nested(0);
#pragma omp parallel for simd shared(c, n) firstprivate(x, y, nums, m) private(i)
for (i = 0; i < n; i++) {
simpleCalcCores(x, c, y, nums, m, i);
}
}
int simpleCheckSplitting(const double *x, const double *c, int *res, int *nums, const int m, const int k, const int id) {
int count = 0, f;
f = getCluster(&x[id * m], &c[0], m, k);
if (f == res[id]) count++;
res[id] = f;
nums[f]++;
return count;
}
char checkSplittingOpenMP(const double *x, const double *c, int *res, int *nums, const int n, const int m, const int k) {
int i, sum = 0;
omp_set_nested(0);
#pragma omp parallel for simd shared(res, nums, n) private(i) firstprivate(x, c, m, k) reduction(+: sum)
for (i = 0; i < n; i++) {
sum += simpleCheckSplitting(x, c, res, nums, m, k, i);
}
return (sum == n) ? 0 : 1;
}
char constr(const int *y, const int val, const int s) {
int i = 0;
while (i < s) {
if (*(y++) == val) return 1;
i++;
}
return 0;
}
void startCoreNums(int *y, const int k, const int n) {
srand((unsigned int)time(NULL));
int i = 0, val;
while (i < k) {
do {
val = rand() % n;
} while (constr(&y[0], val, i));
y[i] = val;
i++;
}
}
void kmeansOpenMP(const double *X, int *y, const int n, const int m, const int k) {
double *x = (double*)malloc(n * m * sizeof(double));
memcpy(x, X, n * m * sizeof(double));
autoscalingOpenMP(x, n, m);
int *nums = (int*)malloc(k * sizeof(int));
startCoreNums(nums, k, n);
double *c = (double*)malloc(k * m * sizeof(double));
detCoresOpenMP(x, c, nums, k, m);
memset(nums, 0, k * sizeof(int));
detStartSplittingOpenMP(x, c, y, nums, n, m ,k);
char flag = 1;
do {
memset(&c[0], 0, k * m * sizeof(double));
calcCoresOpenMP(x, c, y, nums, n, m);
memset(&nums[0], 0, k * sizeof(int));
flag = checkSplittingOpenMP(x, c, y, nums, n, m, k);
} while (flag);
free(nums);
free(c);
free(x);
}
void autoscaling(double *x, const int n, const int m) {
const int s = n * m;
double sd, Ex, Exx;
int i, j = 0;
while (j < m) {
i = j;
Ex = Exx = 0;
while (i < s) {
sd = x[i];
Ex += sd;
Exx += sd * sd;
i += m;
}
Exx /= n;
Ex /= n;
sd = sqrt(Exx - Ex * Ex);
i = j;
while (i < s) {
x[i] = (x[i] - Ex) / sd;
i += m;
}
j++;
}
}
void detCores(const double *x, double *c, const int *sn, const int k, const int m) {
int i;
for (i = 0; i < k; i++) {
memcpy(&c[i * m], &x[sn[i] * m], m * sizeof(double));
}
}
void detStartSplitting(const double *x, const double *c, int *y, int *nums, const int n, const int m, const int k) {
int i = 0, j = 0, cur;
while (i < n) {
cur = getCluster(&x[j], &c[0], m, k);
y[i] = cur;
nums[cur]++;
j += m;
i++;
}
}
void calcCores(const double *x, double *c, const int *res, const int *nums, const int n, const int m) {
int i, j, buf1, buf2, buf3;
for (i = 0; i < n; i++) {
buf1 = nums[res[i]];
buf2 = res[i] * m;
buf3 = i * m;
for (j = 0; j < m; j++) {
c[buf2 + j] += x[buf3 + j] / buf1;
}
}
}
char checkSplitting(const double *x, const double *c, int *res, int *nums, const int n, const int m, const int k) {
int i = 0, count = 0, j = 0, f;
while (i < n) {
f = getCluster(&x[j], &c[0], m, k);
if (f == res[i]) count++;
res[i] = f;
nums[f]++;
j += m;
i++;
}
return (n == count) ? 0 : 1;
}
void kmeans(const double *X, int *y, const int n, const int m, const int k) {
double *x = (double*)malloc(n * m * sizeof(double));
memcpy(x, X, n * m * sizeof(double));
autoscaling(x, n, m);
int *nums = (int*)malloc(k * sizeof(int));
startCoreNums(nums, k, n);
double *c = (double*)malloc(k * m * sizeof(double));
detCores(x, c, nums, k, m);
memset(nums, 0, k * sizeof(int));
detStartSplitting(x, c, y, nums, n, m, k);
char flag = 1;
do {
memset(c, 0, k * m * sizeof(double));
calcCores(x, c, y, nums, n, m);
memset(nums, 0, k * sizeof(int));
flag = checkSplitting(x, c, y, nums, n, m, k);
} while (flag);
free(x);
free(c);
free(nums);
}
