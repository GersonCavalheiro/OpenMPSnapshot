
#include "base_kmean.cpp"

std::vector<Point> assign_closest_cluster(std::vector<Point>& data, std::vector<Point>& centroids){
for (Point& p : data){
double min_dist = DBL_MAX;
for (const Point& c : centroids){
double d = p.compute_distance(c);
if (d < min_dist){
min_dist = d;
p.cluster = c.cluster;
}
}
}
return data;
}


std::vector<Point> kMedoidsClustering(std::vector<Point>(*init_centroids)(const std::vector<Point>&, int&), std::vector<Point>& data, int k, int epochs){
double tstart, tstop;
int count = 0;
std::vector<Point> centroids = init_centroids(data, k);
std::vector<Point> new_centroids = std::vector<Point>(k);

Point partial_new_centroids[omp_get_max_threads()][k];
double partial_min_distances[omp_get_max_threads()][k];

bool converged = false;
tstart = omp_get_wtime();
while (!converged and count < epochs){
data = assign_closest_cluster(data, centroids);

for (int i=0; i<omp_get_max_threads(); i++)
for (int j=0; j<k; j++)
partial_min_distances[i][j] = DBL_MAX;

#pragma omp parallel for num_threads(omp_get_max_threads()) default(none) firstprivate(data) shared(partial_new_centroids, partial_min_distances) schedule(static, 128)
for (auto& p1 : data){
double tot_dist = 0.;
for (auto& p2 : data) {
if (p1.cluster == p2.cluster)
tot_dist += p1.compute_distance(p2);
}
if (tot_dist < partial_min_distances[omp_get_thread_num()][p1.cluster]){
partial_min_distances[omp_get_thread_num()][p1.cluster] = tot_dist;
partial_new_centroids[omp_get_thread_num()][p1.cluster] = p1;
}
}

for (int i = 0; i < k; i++){
double min_d = partial_min_distances[0][i];
new_centroids[i] = partial_new_centroids[0][i];
for (int j = 1; j < omp_get_max_threads(); j++){
if (partial_min_distances[j][i] < min_d){
min_d = partial_min_distances[j][i];
new_centroids[i] = partial_new_centroids[j][i];
}
}
}

converged = true;
for (int i=0; i<k; i++){
if (!(centroids[i] == new_centroids[i])) {
converged = false;
break;
}
}

centroids = new_centroids;
count++;
}

tstop = omp_get_wtime();
printf("Cycles for converging: %d\n", count);
printf("Clustering execution time: %f\n", tstop - tstart);

return centroids;
}