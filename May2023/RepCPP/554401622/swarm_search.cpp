#include "swarm_search.hpp"

#include "test_functions.hpp"
#include "plots.hpp"

#include <cfloat>
#include <cassert>
#include <chrono>
#include <omp.h>

thread_local std::mt19937 SwarmSearch::random_engine;

SwarmSearch::SwarmSearch(
std::function<double(Point)> &objective_func,
size_t n,
size_t particle_count,
int threads,
double min_x,
double max_x
):      objective_func(objective_func),
n(n),
particle_count(particle_count),
threads(threads),
min_x(min_x),
max_x(max_x) {
time_seed = std::chrono::duration_cast< std::chrono::microseconds >(
std::chrono::system_clock::now().time_since_epoch()
).count();
random_engine = std::mt19937(time_seed + threads + 1);
}

void SwarmSearch::setSeed(int thread_id)
{
random_engine = std::mt19937(time_seed + thread_id);
}

SearchResult SwarmSearch::search(size_t iterations)
{
omp_set_num_threads(threads);
auto begin = std::chrono::steady_clock::now();
init();

#pragma omp parallel
{
setSeed(omp_get_thread_num());
}

#pragma omp parallel for shared(best_global_result, c1)
for(size_t i = 0; i < iterations; ++i)
{
for(size_t j = 0; j < particle_count; ++j)
{
updateParticle(particles[j]);
}
c1 *= 0.992;
}

auto end = std::chrono::steady_clock::now();
best_global_result.time = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count();
best_global_result.iterations = iterations;
return best_global_result;
}

SearchResult SwarmSearch::searchForSeconds(int seconds) {
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

SearchResult SwarmSearch::searchUntilStopped() {
omp_set_num_threads(threads);
auto begin = std::chrono::steady_clock::now();
init();

#pragma omp parallel
{
setSeed(omp_get_thread_num());
}

size_t iterations = 0;
#pragma omp parallel shared(force_stop, best_global_result, c1, iterations)
{
while (!force_stop) {
iterations += 1;
for (size_t j = 0; j < particle_count; ++j) {
updateParticle(particles[j]);
}
c1 *= 0.992;
}
}

auto end = std::chrono::steady_clock::now();
best_global_result.time = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count();
best_global_result.iterations = iterations;
return best_global_result;
}

SearchResult SwarmSearch::searchUntilGreaterThan(double threshold) {
omp_set_num_threads(threads);
auto begin = std::chrono::steady_clock::now();
init();

#pragma omp parallel
{
setSeed(omp_get_thread_num());
}

size_t iterations = 0;
#pragma omp parallel shared(best_global_result, c1, iterations)
{
while (best_global_result.result > threshold) {
iterations += 1;
for (size_t j = 0; j < particle_count; ++j) {
updateParticle(particles[j]);
}
c1 *= 0.992;
}
}

auto end = std::chrono::steady_clock::now();
best_global_result.time = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count();
best_global_result.iterations = iterations;
return best_global_result;
}

void SwarmSearch::init()
{
best_global_result.result = DBL_MAX;
for(size_t i = 0; i < n; ++i)
best_global_result.x.push_back(DBL_MAX);

for(size_t i = 0; i < particle_count; ++i)
{
Particle particle;
for(size_t j = 0; j < n; ++j)
{
auto unif = std::uniform_real_distribution<double>(min_x, max_x);
particle.position.push_back(unif(random_engine));
particle.velocity.push_back(0.0);
particle.bestLocalResult.x.push_back(DBL_MAX);
}
particle.bestLocalResult.result = DBL_MAX;
particles.push_back(particle);
}
}

void SwarmSearch::updateParticle(Particle& particle)
{
for(size_t i = 0; i < particle.position.size(); ++i)
{
particle.position[i] += particle.velocity[i];
if(particle.position[i] > 40)
particle.position[i] = 40 - (particle.position[i] - 40); 
if(particle.position[i] < -40)
particle.position[i] = -40 + (-40 - particle.position[i]);
}

double result = objective_func(particle.position);

if(result < particle.bestLocalResult.result)
{
particle.bestLocalResult.result = result;
particle.bestLocalResult.x = particle.position;
}

#ifdef OPENMP_ENABLED
#pragma omp critical
{
#endif
if(result < best_global_result.result)
{
best_global_result.result = result;
best_global_result.x = particle.position;
}
#ifdef OPENMP_ENABLED
}
#endif

updateVelocity(particle);
}

void SwarmSearch::updateVelocity(Particle& particle)
{
for(size_t i = 0; i < particle.velocity.size(); ++i)
{
double r1 = unif01(random_engine), r2 = unif01(random_engine), r3 = unif01(random_engine);
particle.velocity[i] = c1*r1 * particle.velocity[i]
+ c2*r2 * (particle.bestLocalResult.x[i] - particle.position[i])
+ c3*r3 * (best_global_result.x[i] - particle.position[i]);
}
}

void SwarmSearch::plot(size_t iterations, double animation_speed) {
assert(n == 2);
init();

for(size_t i = 0; i < iterations; ++i)
{
plotClear();

for(size_t j = 0; j < particle_count; ++j)
{
Particle& particle = particles[j];

for(size_t k = 0; k < particle.position.size(); ++k)
{
particle.position[k] += particle.velocity[k];
if(particle.position[k] > 40)
particle.position[k] = 40 - (particle.position[k] - 40); 
if(particle.position[k] < -40)
particle.position[k] = -40 + (-40 - particle.position[k]);
}

double result = objective_func(particle.position);

if(result < particle.bestLocalResult.result)
{
particle.bestLocalResult.result = result;
particle.bestLocalResult.x = particle.position;
}

if(result < best_global_result.result)
{
best_global_result.result = result;
best_global_result.x = particle.position;
}

updateVelocity(particle);
}
c1 *= 0.992;

std::cout << "-------------------------------" << std::endl;
std::cout << "iteration: " << i << std::endl;
std::cout << "best_result: " << best_global_result.result << std::endl;
std::cout << "best_position: (" << best_global_result.x[0] << ", " << best_global_result.x[1] << ")" << std::endl;

std::vector<double> x, y, u, v;
for (const auto& particle: particles) {
x.push_back(particle.position[0]);
y.push_back(particle.position[1]);
u.push_back(particle.velocity[0]);
v.push_back(particle.velocity[1]);
}
plotContourWithQuiver(objective_func, best_global_result.x, x, y, u, v, min_x, max_x, animation_speed);
}
}