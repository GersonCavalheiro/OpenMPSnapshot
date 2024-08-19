#include <omp.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <chrono>
#include <random>
#include <assert.h>
#include <cmath>
#include <iomanip>

const auto ARGC = 6;
const auto USAGE_STRING =
"./exec.out print_bool n_points max_dim num_threads hist_buckets\n"
"\nprint_bool\t Prints histograms if print_bool is \"true\"\n"
"n_points \tNumber of points to sample\n"
"max_dim\t Dimensional upper bound. Minimum dim is always 2.\n"
"num_threads\t Number of threads\n"
"hist_buckets\t Number of buckets for the histograms";

const int MIN_DIM = 2;

std::vector<std::vector<int>> generate_histograms(int num_points, int max_dim, int num_threads, int hist_buckets, bool parallel) {
std::vector<std::vector<int>> hists;
std::vector<int> hist(hist_buckets);

std::vector<std::vector<float>> points(num_points, std::vector<float>(max_dim + 2));
std::vector<float> dists(num_points);

std::minstd_rand eng;
std::normal_distribution<float> dist(0, 1);

for (int dims = MIN_DIM; dims <= max_dim; dims++) {
std::fill(dists.begin(), dists.end(), 0);

#pragma omp parallel for num_threads(num_threads) schedule(guided) private(dist, eng, hist) shared(hists, points, dists) if(parallel)
for (std::size_t i = 0; i < points.size(); i++) {
auto &pt = points[i];

float norm = 0;
for (int dim = 0; dim < dims + 2; dim++) {
pt[dim] = dist(eng);
norm += pt[dim] * pt[dim];
}

norm = std::sqrt(norm);

for (int dim = 0; dim < dims + 2; dim++) {
pt[dim] /= norm;
}



for (int dim = 0; dim < dims; dim++) {
dists[i] += pt[dim] * pt[dim];
}
dists[i] = std::sqrt(dists[i]);
}

std::fill(hist.begin(), hist.end(), 0);
for (size_t i = 0; i < dists.size(); i++) {
hist[(int) (dists[i] * hist.size())] += 1;
}

hists.push_back(hist);
}
return hists;
}


void print_hists(std::vector<std::vector<int>> hists, int num_points) {
for (int dims = 0; dims < hists.size(); dims++) {
std::cout << "\n\t" << dims + MIN_DIM << "-D" << std::endl;
std::cout << "Histogram with " << hists[dims].size() << " buckets" << std::endl;
std::cout << "Bucket\t\tPercentage\tCount" << std::endl;
for (size_t i = 0; i < hists[dims].size(); i++) {
std::cout.precision(2);
std::cout << std::fixed
<< ((float) i) / hists[dims].size() << "\t\t"
<< (int)(100.0 * hists[dims][i] / num_points) << "%\t\t"
<< hists[dims][i] << std::endl;
}
}
std::cout << std::endl;
}


void assert_usage(int argc) {
if (argc != ARGC) {
std::cerr << "\nExecution format: " << USAGE_STRING
<< std::endl << std::endl;
exit(1);
}
}


int main(int argc, char **argv) {
assert_usage(argc);

bool print = std::string(argv[1]) == "true";
int n_points = std::stoi(argv[2]);  assert(n_points > 0);
int max_dim = std::stoi(argv[3]);   assert(max_dim >= 2);
int num_threads = std::stoi(argv[4]);
int hist_buckets = std::stoi(argv[5]);

std::cout.precision(8);
std::cout << std::fixed << "\nTiming the histogram generation for distances of "
<< n_points << " points uniformly sampled from the surface of an n-ball, where n ranges from "
<< MIN_DIM << " to " << max_dim << "." << std::endl;
std::cout << "\n\t\tDuration(s)\tMops/s" << std::endl;

generate_histograms(n_points, max_dim, num_threads, hist_buckets, true);

auto start = std::chrono::system_clock::now();
auto hists = generate_histograms(n_points, max_dim, num_threads, hist_buckets, true);
auto stop = std::chrono::system_clock::now();
auto duration = std::chrono::duration<double>(stop - start).count();
std::cout << "Parallel:\t"
<< duration << "\t"
<< n_points / duration / 1'000'000 << std::endl;

start = std::chrono::system_clock::now();
generate_histograms(n_points, max_dim, num_threads, hist_buckets, false);
stop = std::chrono::system_clock::now();
duration = std::chrono::duration<double>(stop - start).count();
std::cout << "Serial:\t\t"
<< duration << "\t"
<< n_points / duration / 1'000'000 << std::endl;

if (print) {
print_hists(hists, n_points);
}
}
