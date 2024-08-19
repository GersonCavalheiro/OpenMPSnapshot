#ifndef KMEAN
#define KMEAN

#include "initializer.cpp"
#include "misc.cpp"

std::vector<Point> kMeansClustering(std::vector<Point>(*init_centroids)(const std::vector<Point>&, int&), std::vector<Point>& data, int k, int epochs, double eps = 0.00001){
double tstart, tstop;
int count = 0;
double rel_diff = 1.;
int clusters_size[k];
std::vector<Point> centroids = init_centroids(data, k);

std::vector<Point> new_centroids = std::vector<Point>(k);
int partial_clusters_size[omp_get_max_threads()][k];
std::vector<Point> partial_new_centroids[omp_get_max_threads()];
double partial_rel_diff[k];

for (int i=0; i<omp_get_max_threads(); i++)
{
partial_new_centroids[i] = std::vector<Point>(k);
for (int j=0; j<k; j++)
{
partial_clusters_size[i][j] = 0;
partial_new_centroids[i][j].to_zero(j);
}
}

int try_num = 1;
tstart = omp_get_wtime();
while (rel_diff > eps){
if (count > epochs){
centroids = init_centroids(data, k); 
count = 0;
try_num++;
tstart = omp_get_wtime();
}

#pragma omp parallel for num_threads(omp_get_max_threads()) default(none) shared(centroids, partial_new_centroids, partial_clusters_size, data) schedule(static, 64)
for(auto & p : data)
{
double best_distance = DBL_MAX;
for (auto &c: centroids) {
double dist = c.compute_distance(p);
if (dist < best_distance) {
best_distance = dist;
p.cluster = c.cluster;
}
}
partial_new_centroids[omp_get_thread_num()][p.cluster] += p;
partial_clusters_size[omp_get_thread_num()][p.cluster]++;
}


#pragma omp parallel for num_threads(k) default(none) shared(new_centroids, partial_new_centroids, partial_clusters_size, partial_rel_diff) firstprivate(centroids, clusters_size, k) schedule(static, 64)
for (int i=0; i < k; i++)
{
new_centroids[i].to_zero(i);
clusters_size[i] = 0;
for (int j = 0; j < omp_get_max_threads(); j++)
{
new_centroids[i] += partial_new_centroids[j][i];
clusters_size[i] += partial_clusters_size[j][i];
partial_new_centroids[j][i].to_zero(i);
partial_clusters_size[j][i] = 0;
}
new_centroids[i] /= clusters_size[i];  
partial_rel_diff[i] = new_centroids[i].compute_rel_diff(centroids[i]);
}

rel_diff = 0.;
for (double& r:partial_rel_diff)
rel_diff += r;
rel_diff /= k;

centroids = new_centroids;
count++;
}

tstop = omp_get_wtime();
printf("Attempt number: %d\n", try_num);
printf("Cycles for converging: %d\n", count);
printf("Clustering execution time: %f\n", tstop - tstart);

return centroids;
}

#endif