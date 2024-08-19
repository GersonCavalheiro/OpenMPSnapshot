#include "Point.h"

std::vector<Point> load_csv(const std::string& filename){
std::vector<Point> data;
double row[FEATS];
std::string line, word;
std::fstream file (filename, std::ios::in);
if(file.is_open())
{
while(getline(file, line))
{
std::stringstream str(line);
int i = 0;
while(getline(str, word, ','))
row[i++] = std::stod(word);
data.emplace_back(row);
}
}
else
std::cout<<"Could not open the file\n";
return data;
}

double compute_silhouette(std::vector<Point>& data, std::vector<Point>& centroids){
unsigned k = centroids.size();
double partial_sse_score[omp_get_max_threads()];
double sse_score;
int cluster_neighbor[k];
std::vector<Point> centroids_points[k];

for (const Point& el: data)
centroids_points[el.cluster].push_back(el);
for (double& el: partial_sse_score)
el = 0.;

for (int i=0; i<k; i++) {
double min_dist = DBL_MAX;
for (int j = 0; j < k; j++) {
double d = centroids[i].compute_distance(centroids[j]);
if (i != j and d < min_dist) {
min_dist = d;
cluster_neighbor[i] = j;
}
}
}

double tstart, tstop;
tstart = omp_get_wtime();
#pragma omp parallel for num_threads(omp_get_max_threads()) default(none) shared(partial_sse_score) firstprivate(data, centroids_points, cluster_neighbor) schedule(static, 64)
for (const Point& p_ref : data){
int c = p_ref.cluster;
double a = 0., b = 0.;
for (const Point& p_same : centroids_points[c]){
a += p_ref.compute_distance(p_same);
}
a /= (double)centroids_points[c].size() - 1;
for (const Point& p_diff : centroids_points[cluster_neighbor[c]]){
b += p_ref.compute_distance(p_diff);
}
b /= (double)centroids_points[c].size();
partial_sse_score[omp_get_thread_num()] += (b - a) / std::max(a, b);
}
tstop = omp_get_wtime();
printf("Silhouette execution time: %f\n", tstop - tstart);

sse_score = std::accumulate(partial_sse_score, partial_sse_score + omp_get_max_threads(), 0.);
sse_score /= (double)data.size();

return sse_score;
}
