#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
#define NUM_THREADS 8
#define TRIALS 1
typedef complex<long double> cmplx;
long double pi = acos(-1);

int bitRev(int in, int log_n) {
int out = 0;
while (log_n--) {
out = (out << 1) | (in & 1);
in >>= 1;
}
return out;
}

void print(const vector<cmplx> & v, string message, bool imag) {
if (v.size() > 16)
return;
cout  << setprecision(6) << message << "\n";
for (int i = 0; i < v.size(); ++i) {
if (imag)
cout << "(" << v[i].real() << ", " << v[i].imag() << ")" << " ";
else
cout << v[i].real() << " ";
}
cout << "\n";
}

vector<cmplx> fft_iterative(vector<cmplx> &in, int inverse) {
int n = in.size();
int log_n = log2(n);
vector<cmplx> a(n);
vector<vector<cmplx>> w_I(log_n+1);

#pragma omp parallel shared(in, a, w_I) firstprivate(n, log_n, inverse)
{
#pragma omp for schedule(static)
for (int i = 0; i < n; ++i) {
int j = bitRev(i, log_n);
a[i] = in[j];
}

#pragma omp for schedule(dynamic)
for (int log_j = 0; log_j <= log_n; log_j++) {

int j = 1 << log_j;
vector<cmplx> w_I_j(j/2+1);

long double theta = 2 * pi / j * (inverse ? -1 : 1);
cmplx w(cos(theta), sin(theta)), w_k(1, 0);

for (int k = 0; k < j/2; ++k) {
w_I_j[k] = w_k;
w_k *= w;
}
w_I[log_j] = w_I_j;
}

for (int log_m = 1; log_m <= log_n; log_m++) {
int m = (1<<log_m);
#pragma omp for schedule(static)
for (int i = 0; i < n; i+=m) {
for (int j = 0; j < m/2; ++j) {
cmplx t1 = a[i+j], t2 = w_I[log_m][j]*a[i+j+m/2];
a[i+j] = t1 + t2;
a[i+j+m/2] = t1 - t2;
}
}
}
}
return a;
}

void ifft_iterative(vector<cmplx> &a){
a = fft_iterative(a, 1);
int n = a.size();
#pragma omp parallel for schedule(static)
for (int i = 0; i < n; ++i)
a[i] /= n;
}



void check(const vector<cmplx> &a, const vector<cmplx> &b) {
double epsilon = 0.00001;
for (int i = 0; i < a.size(); ++i) {
cmplx temp = a[i]-b[i];
if (temp.real() > epsilon || temp.imag() > epsilon) {
cout << "\nArrays don't match" << "\n" << a[i] << " " << b[i] << "\n";
return;
}
}
cout << "\nArrays match" << "\n";
}

int main()
{
omp_set_num_threads(NUM_THREADS);
ifstream f;
f.open("in.txt");
double runtimePy; f >> runtimePy;

int n; f >> n;
vector<cmplx> a(n);
for (int i = 0; i < n; ++i)
f >> a[i];
vector<cmplx> b(a.begin(), a.end());

double runtimeI_FFT, runtimeI_IFFT;

double avg1 = 0;
for (int i = 0; i < TRIALS; ++i) {
runtimeI_FFT = omp_get_wtime();
a = fft_iterative(a, 0);
runtimeI_FFT = omp_get_wtime() - runtimeI_FFT;
avg1 += runtimeI_FFT;
}
avg1 /= TRIALS;
print(a, "Iterative FFT: ", true);

double avg2 = 0;
for (int i = 0; i < TRIALS; ++i) {
runtimeI_IFFT = omp_get_wtime();
ifft_iterative(a);
runtimeI_IFFT = omp_get_wtime() - runtimeI_IFFT;
avg2 += runtimeI_FFT;
}
avg2 /= TRIALS;
print(a, "Iterative Inverse-FFT: ", false);

check(a, b);
cout << "\nRuntime (Average):-" << fixed << "\n";
cout << "Python FFT            :" << runtimePy << "\n";
cout << "Iterative FFT         :" << avg1 << "\n";
cout << "Iterative Inverse FFT :" << avg2 << "\n";
return 0;
}

