#include "Point.h"


bool contains(std::vector<Point>& vec, Point& p){
for (auto & el : vec)
if (p == el)
return true;
return false;
}


std::vector<Point> initialize_centroids_randomly(const std::vector<Point>& data, int& k) {
double tstart, tstop;
tstart = omp_get_wtime();

std::random_device rd;  
std::mt19937 gen(rd()); 
std::uniform_int_distribution<> distrib(0, data.size());
std::vector<Point> centroids;

int clust = 0;
while (centroids.size() < k) {
Point sel_elem = data[distrib(gen)];
if (!contains(centroids, sel_elem))
{
sel_elem.cluster = clust++;
centroids.push_back(sel_elem);
}
}
tstop = omp_get_wtime();
printf("Random Initialization execution time: %f\n", tstop - tstart);

return centroids;
}



std::vector<Point> initialize_centroids_kmeanpp(const std::vector<Point>& data, int& k) {
double tstart, tstop;
tstart = omp_get_wtime();

std::random_device rd;       
std::mt19937 gen(rd());   
std::uniform_int_distribution<> distrib(0, data.size());

std::vector<Point> centroids;
Point starting_point = data[distrib(gen)];
starting_point.cluster = 0;
centroids.push_back(starting_point);

for (int i=1; i<k; i++) {
double max_dist = 0.;
Point next_point;
double partial_max_dist[omp_get_max_threads()];
for (double& el:partial_max_dist)
el = 0.;
Point partial_next_point[omp_get_max_threads()];
#pragma omp parallel for num_threads(omp_get_max_threads()) default(none) shared(partial_max_dist, partial_next_point) firstprivate(centroids, data) schedule(static, 64)
for (const Point& el:data) {
double min_dist = DBL_MAX;
for (const Point& c:centroids) {
double d = el.compute_distance(c);
if (d < min_dist)
min_dist = d;
}
if (min_dist > partial_max_dist[omp_get_thread_num()]){
partial_max_dist[omp_get_thread_num()] = min_dist;
partial_next_point[omp_get_thread_num()] = el;
}
}

for (int j=0; j<omp_get_max_threads(); j++){
if (partial_max_dist[j] > max_dist){
max_dist = partial_max_dist[j];
next_point = partial_next_point[j];
}
}
next_point.cluster = i;
centroids.push_back(next_point);
}
tstop = omp_get_wtime();
printf("KMean++ Initialization execution time: %f\n", tstop - tstart);

return centroids;
}