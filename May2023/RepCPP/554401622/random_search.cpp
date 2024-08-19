#include "random_search.hpp"
#include "test_functions.hpp"
#include "utilities.hpp"
#include "plots.hpp"

#include <cfloat>
#include <cassert>
#include <chrono>
#include <future>
#include <omp.h>

thread_local std::mt19937 RandomSearch::random_engine;

RandomSearch::RandomSearch(
std::function<double(Point)> &objective_func,
size_t n,
int threads,
double min_x,
double max_x
):      objective_func(objective_func),
n(n),
threads(threads),
min_x(min_x),
max_x(max_x) 
{
time_seed = std::chrono::duration_cast< std::chrono::microseconds >(
std::chrono::system_clock::now().time_since_epoch()
).count();

}

void RandomSearch::setSeed(int thread_id)
{
random_engine = std::mt19937(time_seed + thread_id);
}

SearchResult RandomSearch::search(size_t iterations) {
omp_set_num_threads(threads);

auto begin = std::chrono::steady_clock::now();

auto best_result = DBL_MAX;
Point best_position;

auto unifs = getUnifs();

#pragma omp parallel
{
setSeed(omp_get_thread_num());
}

#pragma omp parallel for shared(best_result, best_position, unifs)
for(std::size_t i = 0; i < iterations; ++i)
{
std::vector<double> current_point;
for(auto unif : unifs)
{
current_point.push_back(unif(random_engine));
}

double result = objective_func(current_point);

#ifdef OPENMP_ENABLED
#pragma omp critical
{
#endif
if(result < best_result)
{
best_result = result;
best_position = current_point;
}
#ifdef OPENMP_ENABLED
}
#endif
}

auto end = std::chrono::steady_clock::now();
auto exec_time = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count();

return SearchResult({best_position, best_result, exec_time, iterations});
}

SearchResult RandomSearch::searchUntilStopped() {
omp_set_num_threads(threads);

const auto begin = std::chrono::steady_clock::now();

auto best_result = DBL_MAX;
Point best_position;

auto unifs = getUnifs();

#pragma omp parallel
{
setSeed(omp_get_thread_num());
}

size_t iterations = 0;
#pragma omp parallel shared(force_stop, iterations)
{
while (!force_stop) {
iterations += 1;
std::vector<double> current_point;
for(auto unif : unifs)
{
current_point.push_back(unif(random_engine));
}

double result = objective_func(current_point);

#ifdef OPENMP_ENABLED
#pragma omp critical
{
#endif
if(result < best_result)
{
best_result = result;
best_position = current_point;
}
#ifdef OPENMP_ENABLED
}
#endif
}
}

auto end = std::chrono::steady_clock::now();
auto exec_time = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count();

return SearchResult({best_position, best_result, exec_time, iterations});
}

SearchResult RandomSearch::searchForSeconds(int seconds) {
std::promise<SearchResult> search_result_promise;
auto search_result_future = search_result_promise.get_future();

std::thread worker([&](std::promise<SearchResult>&& search_result_promise){
auto search_result = this->searchUntilStopped();
search_result_promise.set_value(search_result);
}, std::move(search_result_promise));

std::thread supervisor([&](){
auto begin = std::chrono::steady_clock::now();
while (true) {
auto end = std::chrono::steady_clock::now();
double exec_time = std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() / 1000.0;
if (exec_time > seconds) {
this->forceStop();
break;
}
}
});

worker.join();
supervisor.join();

return search_result_future.get();
}

SearchResult RandomSearch::searchUntilGreaterThan(double threshold) {
omp_set_num_threads(threads);

const auto begin = std::chrono::steady_clock::now();

auto best_result = DBL_MAX;
Point best_position;

auto unifs = getUnifs();

#pragma omp parallel
{
setSeed(omp_get_thread_num());
}

size_t iterations = 0;
#pragma omp parallel shared(iterations)
{
while (best_result > threshold) {
iterations += 1;
std::vector<double> current_point;
for(auto unif : unifs)
{
current_point.push_back(unif(random_engine));
}

double result = objective_func(current_point);

#ifdef OPENMP_ENABLED
#pragma omp critical
{
#endif
if(result < best_result)
{
best_result = result;
best_position = current_point;
}
#ifdef OPENMP_ENABLED
}
#endif
}
}

auto end = std::chrono::steady_clock::now();
auto exec_time = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count();

return SearchResult({best_position, best_result, exec_time, iterations});
}

Unifs RandomSearch::getUnifs() {
Unifs unifs;
for(size_t i = 0; i < n; ++i)
{
unifs.emplace_back(min_x, max_x);
}
return unifs;
}

void RandomSearch::plot(size_t iterations, double animation_speed) {
assert(n == 2);

auto best_result = DBL_MAX;
Point best_position;

std::vector<std::uniform_real_distribution<double>> unifs {
std::uniform_real_distribution<double>(min_x, max_x),
std::uniform_real_distribution<double>(min_x, max_x)
};

for(std::size_t i = 0; i < iterations; ++i)
{
plotClear();

std::vector<double> random_point;
for(auto unif : unifs)
{
random_point.push_back(unif(random_engine));
}

double result = objective_func(random_point);

if(result < best_result)
{
best_result = result;
best_position = random_point;
}

std::cout << "-------------------------------" << std::endl;
std::cout << "iteration: " << i << std::endl;
std::cout << "best_result: " << best_result << std::endl;
std::cout << "best_position: (" << best_position[0] << ", " << best_position[1] << ")" << std::endl;

plotContourWithBestAndCurrentPoint(objective_func, best_position, random_point, min_x, max_x, animation_speed);
}
}



