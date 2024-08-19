
#include "integral.h"

double Integral::calculate(vector<double> y_values, int dots_number, double partition, int threads) {
double total_sum = 0;
auto start = chrono::system_clock::now();
#pragma omp parallel for shared(y_values, dots_number, partition) reduction(+:total_sum) default(none) num_threads(threads)
{
for (int i = 0; i < dots_number; i++) {
total_sum = total_sum + y_values[i] * partition;
}
}
auto end = chrono::system_clock::now();
long int total_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
printf("Integral of sin(x): %f\nExecution time: %ld ms\n", total_sum, total_time);
return total_sum;
}

vector<double> Integral::get_y_values(double segment_length, double segment_end, int dots_number, double partition) {
vector<double> y_values = {};
vector<double> x_values = {segment_end};
printf("partition: %f\ninterval: [%f: %f]\n", partition, x_values[0], x_values[0] + segment_length);
y_values.reserve(dots_number);
for (int i = 0; i < dots_number; i++) {
y_values.push_back(sin(x_values[i]));
x_values.push_back(x_values[i] + partition);
}
return y_values;
}