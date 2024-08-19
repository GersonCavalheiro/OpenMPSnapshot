#define NORMAL_DISTRIBUTION_OPENMP
#ifdef NORMAL_DISTRIBUTION_OPENMP



#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define NUM_SAMPLES 1000u


#define NUM_LOOPS 1000u






void normal_by_definition(const double mean, const double std, double *arr, const unsigned length) {
const double two_pi = 8.0 * atan(1.0);
int i;

#pragma omp parallel for default(none) private(i) shared(mean, std, arr, length, two_pi) schedule(static, 10)
for (i = 0; i < length; i++) {
double x = rand() * 1. / RAND_MAX;
double y = (1. / (std * sqrt(two_pi))) * (exp(-(x - mean)*(x - mean) / (2. * std*std)));
arr[i] = y * std + mean;
}
}



void normal_clt(const double mean, const double std, double *arr, const unsigned length) {
const unsigned n_sum = 25;
int i;

#pragma omp parallel for default(none) private(i) shared(mean, std, arr, length, n_sum) schedule(static, 10)
for (i = 0; i < length; i++) {
double s = 0;
for (size_t j = 0; j < n_sum; j++) {
s += (double)rand() / RAND_MAX;
}
s -= n_sum / 2.0;
s /= sqrt(n_sum / 12.0);
arr[i] = s * std + mean;
}
}


void normal_box_muller_slow(const double mean, const double std, double *arr, const unsigned length) {
const double two_pi = 8.0 * atan(1.0);
double x1;
double x2;
int i;

#pragma omp parallel for default(none) private(i) shared(mean, std, arr, length, two_pi, x1, x2) schedule(static)
for (i = 0; i < length; i++) {
double y;
if (i % 2 == 0) {
x1 = (rand() + 1.) / (RAND_MAX + 2.);
x2 = rand() / (RAND_MAX + 1.);
y = sqrt(-2.0 * log(x1)) * sin(two_pi * x2);
}
else {
y = sqrt(-2.0 * log(x1)) * cos(two_pi * x2);
}
arr[i] = y * std + mean;
}
}



void normal_box_muller_fast(const double mean, const double std, double *arr, const unsigned length) {
const double two_pi = 8.0 * atan(1.0);
int i;

#pragma omp parallel for default(none) private(i) shared(mean, std, arr, length, two_pi) schedule(runtime)
for (i = 0; i < length; i += 2) {
double x1, x2, y1, y2;
x1 = (rand() + 1.) / (RAND_MAX + 2.);
x2 = rand() / (RAND_MAX + 1.);
y1 = sqrt(-2.0 * log(x1)) * sin(two_pi * x2);
y2 = sqrt(-2.0 * log(x1)) * cos(two_pi * x2);
arr[i] = y1 * std + mean;
arr[i + 1] = y2 * std + mean;
}
}


void normal_marsaglia(const double mean, const double std, double *arr, const unsigned length) {
int i;

#pragma omp parallel for default(none) private(i) shared(mean, std, arr, length) schedule(static, 10)
for (i = 0; i < length; i += 2) {
double x1, x2, s, y1, y2, f;
do {
x1 = 2.0 * rand() / (double)RAND_MAX - 1.0;
x2 = 2.0 * rand() / (double)RAND_MAX - 1.0;
s = x1 * x1 + x2 * x2;
} while (s >= 1.0 || s == 0.0);
f = sqrt(-2.0 * log(s) / s);
y1 = x1 * f;
y2 = x2 * f;
arr[i] = y1 * std + mean;
arr[i + 1] = y2 * std + mean;
}
}





double mean(const double *arr, const unsigned length) {
double sum = 0.;
int i;

#pragma omp parallel for default(none) private(i) shared(arr, length) reduction(+:sum) schedule(static)
for (i = 0; i < length; i++) {
sum += arr[i];
}

return sum / length;
}


double std_dev(const double *arr, const unsigned length) {
double avg = mean(arr, length);
double sum = 0.;
int i;

#pragma omp parallel for default(none) private(i) shared(arr, length, avg) reduction(+:sum) schedule(static)
for (i = 0; i < length; i++) {
sum += (arr[i] - avg) * (arr[i] - avg);
}

return sqrt(sum / (length - 1));
}


void print_array(double *arr, const unsigned length, const unsigned num_to_print) {
if (num_to_print > length) {
printf("num_to_print has to be <= size!\n");
system("pause");
exit(-1);
}

for (size_t i = 0; i < num_to_print; ++i) {
printf("%f ", arr[i]);
}
puts("");
}


void generate_and_print(const char *function_name, void(*fp)(const double, const double, double*, const unsigned), \
const double mean_, const double std, double *arr, const unsigned length, const unsigned num_to_print) {

(*fp)(mean_, std, arr, length);

printf("\n%s:\n", function_name);
printf("The first %u elements:\n", num_to_print);
print_array(arr, length, num_to_print);

double avg = mean(arr, length);
double stddev = std_dev(arr, length);
printf("Mean: %f; Standard deviation: %f\n", avg, stddev);
printf("\tAbsolute error for mean: %f; Absolute error for standard deviation: %f\n", fabs(avg - mean_), fabs(stddev - std));
}

void measure_time(const char *function_name, void(*fp)(const double, const double, double*, const unsigned), \
const double mean, const double std, double *arr, const unsigned length, const unsigned num_loops) {


double t0, t1;

t0 = omp_get_wtime();

for (size_t i = 0; i < num_loops; ++i) {
(*fp)(mean, std, arr, length);
}

t1 = omp_get_wtime();
printf("%s took %.3f s\n", function_name, t1 - t0);
}

int main(int argc, char *argv[]) {

time_t t;
srand((unsigned)time(&t));


const unsigned num_samples = NUM_SAMPLES;
double *samples = malloc(num_samples * sizeof(*samples));


void(*fp)(const double, const double, double*, const unsigned) = NULL;


const double mean = -100.0;
const double std = 10.0;

const unsigned num_loops = NUM_LOOPS;

measure_time("normal_by_definition", normal_by_definition, mean, std, samples, num_samples, num_loops);
measure_time("normal_clt", normal_clt, mean, std, samples, num_samples, num_loops);
measure_time("normal_box_muller_slow", normal_box_muller_slow, mean, std, samples, num_samples, num_loops);
measure_time("normal_box_muller_fast", normal_box_muller_fast, mean, std, samples, num_samples, num_loops);
measure_time("normal_marsaglia", normal_marsaglia, mean, std, samples, num_samples, num_loops);

const unsigned num_to_print = 10;

generate_and_print("normal_by_definition", normal_by_definition, mean, std, samples, num_samples, num_to_print);
generate_and_print("normal_clt", normal_clt, mean, std, samples, num_samples, num_to_print);
generate_and_print("normal_box_muller_slow", normal_box_muller_slow, mean, std, samples, num_samples, num_to_print);
generate_and_print("normal_box_muller_fast", normal_box_muller_fast, mean, std, samples, num_samples, num_to_print);
generate_and_print("normal_marsaglia", normal_marsaglia, mean, std, samples, num_samples, num_to_print);

free(samples);

puts("");
system("pause");
return(0);
}

#endif 

