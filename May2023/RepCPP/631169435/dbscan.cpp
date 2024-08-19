#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <limits>
#include <algorithm>
#include <numeric>
#include <functional>
#include <chrono>
#include <queue>
#include <future>
#include <deque>
#include <immintrin.h>
#include "utils.h"
#include "point.h"
#include "dbscan.h"
#include <omp.h>


[[gnu::const]] static bool anyDistWithinEps(__m256d lhs_x, __m256d lhs_y, __m256d rhs_x, __m256d rhs_y, __m256d eps_AVX) {
const __m256d diff_x = _mm256_sub_pd(lhs_x, rhs_x);
const __m256d diff_y = _mm256_sub_pd(lhs_y, rhs_y);
const __m256d sqr_x = _mm256_mul_pd(diff_x, diff_x);
const __m256d sqr_y = _mm256_mul_pd(diff_y, diff_y);
const __m256d sum = _mm256_add_pd(sqr_x, sqr_y);
const __m256d dist_squared = _mm256_sqrt_pd(sum);
const __m256d cmp_result = _mm256_cmp_pd(dist_squared, eps_AVX, _CMP_LT_OS);
const int mask = _mm256_movemask_pd(cmp_result);
return mask != 0;
}


static void alignAVXbuffer(std::vector<Point> & vp) {   
while ((vp.size() % 4) != 0) {
const Point copyPoint{vp[vp.size()-1]};
vp.push_back(copyPoint);
}
}

template <typename Iterator1, typename Iterator2>
[[gnu::const]] static bool isConnect_AVX(Iterator1 lhs_begin, Iterator1 lhs_end, Iterator2 rhs_begin, Iterator2 rhs_end, double eps) {

const __m256d eps_AVX = _mm256_set1_pd(eps);
__m256d lhs_x{};
__m256d lhs_y{};
__m256d rhs_x{};
__m256d rhs_y{};
for (auto lhs_it = lhs_begin; lhs_it != lhs_end; ++lhs_it) {
lhs_x = _mm256_set1_pd((*lhs_it)[0]);
lhs_y = _mm256_set1_pd((*lhs_it)[1]);
for (auto rhs_it = rhs_begin; rhs_it < rhs_end; rhs_it += 4) {
rhs_x = _mm256_set_pd((*std::next(rhs_it, 3))[0], (*std::next(rhs_it, 2))[0], (*std::next(rhs_it, 1))[0], (*rhs_it)[0]);
rhs_y = _mm256_set_pd((*std::next(rhs_it, 3))[1], (*std::next(rhs_it, 2))[1], (*std::next(rhs_it, 1))[1], (*rhs_it)[1]);
if (anyDistWithinEps(lhs_x, lhs_y, rhs_x, rhs_y, eps_AVX))
return true;
}
}
return false;
}


std::vector<int> NaiveDBSCAN::dbscan_algorithm(std::vector<Point> const& points) {
std::cout << "Naive DBSCAN(eps="<<eps<<", minPts="<<minPts<<") on datasize: " << points.size() << " in " << Point::dimensionality<<"-dimension space" << '\n';
const auto start = std::chrono::high_resolution_clock::now();
std::vector<int> visited(points.size(), 0);
std::vector<int> cluster(points.size(), -1);
int clusterIdx{0};
std::vector<int> neighborIdxs{};

for (int i = 0; i < points.size(); i++) {
if (visited[i] == 1) continue;

visited[i] = 1;
neighborIdxs = rangeQuery(points, i, eps);
if (neighborIdxs.size() >= minPts) {
cluster[i] = clusterIdx;
for (int j = 0; j < neighborIdxs.size(); j++) {
int idx = neighborIdxs[j];
if (visited[idx] == 1) continue;
visited[idx] = 1;
if (cluster[idx] == -1) {
cluster[idx] = clusterIdx;
}
std::vector<int> subNeighborIdxs = rangeQuery(points, idx, eps);
if (subNeighborIdxs.size() >= minPts) {
neighborIdxs.insert(neighborIdxs.end(), subNeighborIdxs.begin(), subNeighborIdxs.end());
}
}
clusterIdx++;
}
}
auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed = end - start;
std::cout << "Elapsed time: " << elapsed.count() << " seconds." << '\n';
return cluster;
}

[[gnu::const]] std::vector<int> getGridSize(std::vector<double> const& max_values, std::vector<double> const& min_values, double gridCellSize) {
std::vector<int> gridSize(max_values.size());
for (int i = 0; i < max_values.size(); i++) {
gridSize[i] = static_cast<int>(std::ceil((max_values[i] - min_values[i]) / gridCellSize));
}
return gridSize;
}

[[gnu::const]] static int kDTo1DIdx(std::vector<int> const& index, std::vector<int> const& gridSize) {
const int k{index.size()};
int Idx1D{index[0]};
int stride{1};
for (int i = 1; i < k; i++) {
stride *= gridSize[i-1];
Idx1D += stride * index[i];
}
return Idx1D;
}

[[gnu::const]] static std::vector<int> oneDToKDIdx(int index, std::vector<int> const& gridSize) {
std::vector<int> kDIdx(gridSize.size(), 0);
const int k{gridSize.size()};
for (int i = k - 1; i >= 0; i--) {
kDIdx[i] = index % gridSize[i];
index /= gridSize[i];
}
return kDIdx;
}

[[gnu::pure]] static std::vector<int> getNeighborIndices(std::vector<int> const& index, std::vector<int> const& gridSize) {
std::vector<int> res{};
if (Point::dimensionality == 1) {
res = {index[0]-1, index[0]+1};
} else if (Point::dimensionality == 2) {
for (int i = std::max(0,index[0]-2); i < std::min(index[0]+3, gridSize[0]); i++) {
for (int j = std::max(0,index[1]-2); j < std::min(index[1]+3, gridSize[0]); j++) {
if ((abs(i-index[0])+abs(j-index[1])) == 4) continue;
if ((i == index[0]) && (j == index[1])) continue;
res.push_back((i)*gridSize[0]+(j));
}
}
}
return res;
}

[[gnu::const]] static bool isConnect(std::vector<Point> lhs, std::vector<Point> rhs, double eps) {
for (auto const& lhsp: lhs)
for(auto const& rhsp: rhs)
if (dist(lhsp, rhsp) <= eps)
return true;
return false;
}

[[gnu::const]] static int getConnectCount(Point lhsp, std::vector<Point> const& rhs, double eps) {
return std::count_if(rhs.begin(), rhs.end(), [&](Point const& rhsp) {
return dist(lhsp, rhsp) <= eps;
});
}

GridDBSCAN::GridDBSCAN(double _eps, int _minPts, std::string _className)
:   DBSCAN(_eps, _minPts), 
corecell_set(std::vector<int>()),
className(_className)
{

}

void GridDBSCAN::assignPoints(std::vector<Point> const& points) {
std::vector<int> index(Point::dimensionality);
int GridIndex1D{};
const double _gridCellSize = gridCellSize;
for (auto const& p : points) {
std::transform(p.begin(), p.end(), index.begin(), [_gridCellSize](double x) {
return static_cast<int>(x / _gridCellSize);
});
GridIndex1D = kDTo1DIdx(index, gridSize);
grid[GridIndex1D].push_back(p);
}
}

void GridDBSCAN::mark_ingrid_corecell() {
for (int i = 0; i<gridSize1D; i++) {
if (grid[i].size() >= minPts) {
corecell[i] = true;
corecell_set.push_back(i);
} 
}
}

void GridDBSCAN::mark_outgrid_corecell() {
for (int i = 0; i<gridSize1D; i++) {
if (corecell[i] == true) continue;
if (grid[i].size() > 0)
corecell[i] = mark_outgrid_corecell_helper(i);
}
}

void SerialGridDBSCAN::expand() {
for (int _i = 0; _i < corecell_set.size(); _i++) {
expand_helper(corecell_set[_i]);
}
}

void OMPGridDBSCAN::expand() {
#pragma omp parallel for shared(uf, grid)
for (int _i = 0; _i < corecell_set.size(); _i++) {
expand_helper(corecell_set[_i]);
}
}

int get_number_of_threads() {
const char* env_threads = std::getenv("OMP_NUM_THREADS");
return (env_threads != nullptr) ? std::stoi(env_threads) : std::thread::hardware_concurrency();
}

void ConcurrencyGridDBSCAN::expand() {
const int num_threads(get_number_of_threads());

std::vector<std::future<void>> futures;
futures.reserve(num_threads);

const int last{corecell_set.size()};
const int blockSize{last/num_threads};

int start{0};

using helper_func_t = void (ConcurrencyGridDBSCAN::*)(int, int);
helper_func_t helper_func_ptr = &ConcurrencyGridDBSCAN::expand_helper;
for (int _i = 0; _i < num_threads-1; _i++) {
futures.push_back(std::async(std::launch::async, helper_func_ptr, this, start, start+blockSize));
start += blockSize;
}
futures.push_back(std::async(std::launch::async, helper_func_ptr, this, start, last));

for (auto &future : futures) {
future.wait();
}
}

std::vector<std::mutex> work_queue_mutex;

void ConcurrencyStealingGridDBSCAN::expand() {
const int num_threads = get_number_of_threads();

std::vector<std::future<void>> futures;
futures.reserve(num_threads);

work_queue_mutex = std::vector<std::mutex>(num_threads);

std::deque<int> tasks;
tasks.resize(corecell_set.size());
std::copy(corecell_set.begin(), corecell_set.end(), tasks.begin());

std::vector<std::deque<int>> work(num_threads);
for (int i = 0; i < corecell_set.size(); i++) {
work[i % num_threads].push_back(tasks.front());
tasks.pop_front();
}

using helper_func_t = void (ConcurrencyStealingGridDBSCAN::*)(std::deque<int>*, int, std::deque<int>*, int);
helper_func_t helper_func_ptr = &ConcurrencyStealingGridDBSCAN::expand_helper;
for (int i = 0; i < num_threads; i++) {
futures.push_back(std::async(std::launch::async, helper_func_ptr, this, &work[i], i, &work[(i+1)%num_threads], (i+1)%num_threads));
}

for (auto &future : futures) {
future.wait();
}
}

std::vector<int> GridDBSCAN::getClusterResults(std::vector<Point> const& points) {
std::vector<int> pointsCluster(npoints, -1);
std::vector<int> index(Point::dimensionality);
const double _gridCellSize{gridCellSize};
for (auto const& p: points) {
std::transform(p.coords.begin(), p.coords.end(), index.begin(), [_gridCellSize](double x) {
return static_cast<int>(x / _gridCellSize);
});
pointsCluster[p.id] = uf.find(kDTo1DIdx(index, gridSize));
}
return pointsCluster;
}

std::vector<int> GridDBSCAN::findNeighbor(int i) {
const std::vector<int> kDIndex(oneDToKDIdx(i, gridSize));
const std::vector<int> neighborsIn1D(getNeighborIndices(kDIndex, gridSize));
return neighborsIn1D;
}

bool GridDBSCAN::mark_outgrid_corecell_helper(int i) {
const std::vector<int> neighbors=findNeighbor(i);
int numConn{0};
for (auto const& g: grid[i]) {
numConn=0;
for (auto const& nid: neighbors) {
numConn += getConnectCount(g, grid[nid], eps);
if (numConn >= minPts) {
corecell_set.push_back(i);
return true;
}
}
}
return false;
}

void GridDBSCAN::_expand_helper(int i) {
std::queue<int> q;
q.push(i);
int node{};
std::vector<int> neighbors{};
while (!q.empty()) {
node = q.front();
q.pop();
neighbors=findNeighbor(node);
for (auto const& ni:neighbors) {
if (visited[ni] == true) continue;
if (isConnect(grid[i], grid[ni], eps)) {
visited[ni] = true;
cluster[ni] = clusterIdx;
if (corecell[ni]==true)
q.push(ni);
}
}
}
}

void GridDBSCAN::expand_helper(int i) {
uf.find(i);
std::vector<int> neighbors=findNeighbor(i);
for (auto const& ni:neighbors) {
if (i <= ni) continue;
if (isConnect(grid[i], grid[ni], eps)) {
uf.unite(i, ni);
}
}
}

void ConcurrencyGridDBSCAN::expand_helper(int lo, int hi) {
int i{};
std::vector<int> neighbors{};
for (int _i = lo; _i < hi; _i++) {
i = corecell_set[_i];
neighbors = findNeighbor(i);
for (auto const& ni:neighbors) {
if (i <= ni) continue;
if (isConnect(grid[i], grid[ni], eps)) {
uf.unite(i, ni);
}
}
}
}

void ConcurrencyStealingGridDBSCAN::expand_helper(std::deque<int> *cells, int tid, std::deque<int> *neighbor, int nid) {
int cell_index{};
std::vector<int> neighbors{};
while (!cells->empty()) {
std::unique_lock<std::mutex> lock(work_queue_mutex[tid]);
if (cells->empty()) {
lock.unlock();
break;
}
cell_index = cells->front();
cells->pop_front();
lock.unlock();

uf.find(cell_index);
neighbors = findNeighbor(cell_index);
for (auto const& neighbor_index : neighbors) {
if (cell_index <= neighbor_index) continue;
if (isConnect(grid[cell_index], grid[neighbor_index], eps)) {
uf.unite(cell_index, neighbor_index);
}
}
}
while (!neighbor->empty()) {
std::unique_lock<std::mutex> lock(work_queue_mutex[nid]);
if (neighbor->empty()) {
lock.unlock();
break;
}
cell_index = neighbor->front();
neighbor->pop_front();
lock.unlock();

uf.find(cell_index);
neighbors = findNeighbor(cell_index);
for (auto const& neighbor_index : neighbors) {
if (cell_index <= neighbor_index) continue;
if (isConnect(grid[cell_index], grid[neighbor_index], eps)) {
uf.unite(cell_index, neighbor_index);
}
}
}
}

std::vector<Point> GridDBSCAN::preprocess(std::vector<Point> const& points) {
gridCellSize = eps / sqrt(Point::dimensionality);
const auto& [max_values, min_values] = calculateMinMaxValues(points);
gridSize = getGridSize(max_values, min_values, gridCellSize);
gridSize1D = std::accumulate(gridSize.begin(), gridSize.end(), 1, std::multiplies<int>());
grid.resize(gridSize1D);
corecell.resize(gridSize1D, false);
visited.resize(gridSize1D, 0);
cluster.resize(gridSize1D, -1);
clusterIdx = 0;
npoints = points.size();
uf.setup(gridSize1D);
return points;
}

std::vector<int> GridDBSCAN::dbscan_algorithm(std::vector<Point> const& points) {
if (Point::dimensionality > 2) {
std::cout << "Skip. Only support 2 dimension "<< className <<" in this project." << '\n';
return std::vector<int>(points.size(), -1);
}

std::cout << className <<"(eps="<<eps<<", minPts="<<minPts<<") on datasize: " << points.size() << " in " << Point::dimensionality<<"-dimension space" << '\n';
const auto start = std::chrono::high_resolution_clock::now();

auto _start = std::chrono::high_resolution_clock::now();
assignPoints(points);
auto _end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> _elapsed = _end - _start;
std::cout << className <<" - assignPoints Elapsed time: " << _elapsed.count() << " seconds." << '\n';

_start = std::chrono::high_resolution_clock::now();
mark_ingrid_corecell();
_end = std::chrono::high_resolution_clock::now();
_elapsed = _end - _start;
std::cout << className <<" - mark_ingrid_corecell  time: " << _elapsed.count() << " seconds." << '\n';

_start = std::chrono::high_resolution_clock::now();
mark_outgrid_corecell();
_end = std::chrono::high_resolution_clock::now();
_elapsed = _end - _start;
std::cout << className <<" - mark_outgrid_corecell Elapsed time: " << _elapsed.count() << " seconds." << '\n';

std::cout << className <<" - corecell_set: " << corecell_set.size() << '\n';

_start = std::chrono::high_resolution_clock::now();
expand();
_end = std::chrono::high_resolution_clock::now();
_elapsed = _end - _start;
std::cout << className <<" - expand Elapsed time: " << _elapsed.count() << " seconds." << '\n';

const auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed = end - start;
std::cout << "Elapsed time: " << elapsed.count() << " seconds." << '\n';

std::vector<int> pointsCluster = getClusterResults(points);
return pointsCluster;
}

void print_clusters(std::vector<Point> const& points, std::vector<int> const& cluster) {
for (int i = 0; i < points.size(); i++) {
std::cout << "(";
for (int j = 0; j < points[i].coords.size(); j++) {
std::cout << points[i].coords[j];
if (j < points[i].coords.size() - 1) {
std::cout << ", ";
}
}
std::cout << "): " << cluster[i] << '\n';
}
}

void calculateAccuracy(std::vector<Point> const& points, std::vector<int> const& cluster) {
if (points.size() != cluster.size()) [[unlikely]] {
std::cout << "Unmatched size\n";
return;
}
const int n{static_cast<int>(points.size())};
int correct{0};
for (int i = 0; i < n; i++) {
if (points[i].label == cluster[i]) {
correct++;
}
}
if (n <= 0) [[unlikely]] {
std::cout << "No accuracy data due to 'Div by zero'. Please make sure the input data is not empty.\n" ;
} else {
std::cout << "Accuracy: " << static_cast<double>(correct)/n << "\n";
}
}

void DBSCAN::run(std::string filename) {
Point::resetDimension();
auto points(preprocess(normalize(parseDataset(filename))));
auto cluster(dbscan_algorithm(points));
calculateAccuracy(points, cluster);
}

void DBSCAN::run(std::vector<std::vector<double>> data) {
Point::resetDimension();
auto points(preprocess(normalize(parseRandomGeneratedData(data))));
auto cluster(dbscan_algorithm(points));
}

OMPGridDBSCAN::OMPGridDBSCAN(double _eps, int _minPts)
: GridDBSCAN(_eps, _minPts, "OMPGridDBSCAN") 
{
#pragma omp parallel 
{
#pragma omp single 
{
std::cout << "Number of threads: " << omp_get_num_threads() << '\n';
}
}
}


ConcurrencyGridDBSCAN::ConcurrencyGridDBSCAN(double _eps, int _minPts)
: GridDBSCAN(_eps, _minPts, "ConcurrencyGridDBSCAN") 
{
std::cout << "Number of threads: " << get_number_of_threads() << "\n";
}

ConcurrencyStealingGridDBSCAN::ConcurrencyStealingGridDBSCAN(double _eps, int _minPts)
: GridDBSCAN(_eps, _minPts, "ConcurrencyStealingGridDBSCAN") 
{
std::cout << "Number of threads: " << get_number_of_threads() << "\n";
}

ConcurrencyStealingAVX2GridDBSCAN::ConcurrencyStealingAVX2GridDBSCAN(double _eps, int _minPts)
: GridDBSCAN(_eps, _minPts, "ConcurrencyStealingAVX2GridDBSCAN")
{
std::cout << "Number of threads: " << get_number_of_threads() << "\n";
}

void ConcurrencyStealingAVX2GridDBSCAN::expand() {
const int num_threads = get_number_of_threads();

std::vector<std::future<void>> futures;
futures.reserve(num_threads);

work_queue_mutex = std::vector<std::mutex>(num_threads);

std::deque<int> tasks;
for (int i = 0; i < corecell_set.size(); i++) {
tasks.push_back(corecell_set[i]);
}

std::vector<std::deque<int>> work(num_threads);
for (int i = 0; i < corecell_set.size(); i++) {
work[i % num_threads].push_back(tasks.front());
tasks.pop_front();
}

for (int i = 0; i < num_threads; i++) {
using helper_func_t = void (ConcurrencyStealingAVX2GridDBSCAN::*)(std::deque<int>*, int, std::deque<int>*, int);
helper_func_t helper_func_ptr = &ConcurrencyStealingAVX2GridDBSCAN::expand_helper;
futures.push_back(std::async(std::launch::async, helper_func_ptr, this, &work[i], i, &work[(i+1)%num_threads], (i+1)%num_threads));
}

for (auto &future : futures) {
future.wait();
}
}

void ConcurrencyStealingAVX2GridDBSCAN::expand_helper(std::deque<int> *cells, int tid, std::deque<int> *neighbor, int nid) {
int cell_index{};
std::vector<int> neighbors{};
while (!cells->empty()) {
std::unique_lock<std::mutex> lock(work_queue_mutex[tid]);
if (cells->empty()) {
lock.unlock();
break;
}
cell_index = cells->front();
cells->pop_front();
lock.unlock();

uf.find(cell_index);
neighbors = findNeighbor(cell_index);
for (auto const& neighbor_index : neighbors) {
if (cell_index <= neighbor_index) continue;
alignAVXbuffer(grid[neighbor_index]);
if (isConnect_AVX(grid[cell_index].begin(), grid[cell_index].end(),
grid[neighbor_index].begin(), grid[neighbor_index].end(), eps)) {
uf.unite(cell_index, neighbor_index);\
}
}
}
while (!neighbor->empty()) {
std::unique_lock<std::mutex> lock(work_queue_mutex[nid]);
if (neighbor->empty()) {
lock.unlock();
break;
}
cell_index = neighbor->front();
neighbor->pop_front();
lock.unlock();

uf.find(cell_index);
neighbors = findNeighbor(cell_index);
for (auto const& neighbor_index : neighbors) {
if (cell_index <= neighbor_index) continue;
alignAVXbuffer(grid[neighbor_index]);
if (isConnect_AVX(grid[cell_index].begin(), grid[cell_index].end(),
grid[neighbor_index].begin(), grid[neighbor_index].end(), eps)) {
uf.unite(cell_index, neighbor_index);
}
}
}
}