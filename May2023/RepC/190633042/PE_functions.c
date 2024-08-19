#include "PE_functions.h"
int read_graph_from_file(char *filename, double **val, int **col_idx, int **row_ptr, int **D, int *dangling_count){
FILE *infile = fopen(filename,"r");
if(infile==NULL){
printf("File not found.\n");
exit(0);
}
printf("\n******************************\n");
printf("Web graph: %s\n", filename);
int node_count;
int edge_count;
for(int i=0; i<2; i++) fscanf(infile, "%*[^\n]\n"); 
fscanf(infile, "%*s %*s %d %*s %d ", &node_count, &edge_count); 
fscanf(infile, "%*[^\n]\n"); 
printf("Nodes: %d, edges: %d\n", node_count, edge_count);
(*row_ptr) = malloc((node_count+1)*sizeof*(*row_ptr));
int *outbound_count = calloc(node_count, sizeof*outbound_count);
int *inbound_count = calloc(node_count, sizeof*inbound_count);
int *from_node_id = malloc(edge_count*sizeof*from_node_id);
int *to_node_id = malloc(edge_count*sizeof*to_node_id);
int edge_count_new = 0;
int from_node;
int to_node;
for(int edge=0; edge<edge_count; edge++){
fscanf(infile, "%d %d", &from_node, &to_node);
if(from_node != to_node){
from_node_id[edge_count_new] = from_node;
to_node_id[edge_count_new] = to_node;
outbound_count[from_node]++;
inbound_count[to_node]++;
edge_count_new++;
}
}
fclose(infile);
printf("Number of self-links: %d\n", edge_count - edge_count_new);
edge_count = edge_count_new; 
(*val) = malloc(edge_count*sizeof*(*val));
(*col_idx) = malloc(edge_count*sizeof*(*col_idx));
int sum = 0;
(*row_ptr)[0] = 0;
for(int i=1; i<node_count+1; i++){
sum += inbound_count[i-1];
(*row_ptr)[i] = sum;
}
for(int node=0; node<node_count; node++){
if(outbound_count[node]==0) (*dangling_count)++;
}
if((*dangling_count)>0){
printf("Number of dangling webpages: %d\n", *dangling_count);
(*D) = malloc(*dangling_count*sizeof*(*D));
int node = 0;
for(int d_node=0; d_node<*dangling_count; d_node++){
while(outbound_count[node] != 0){
node++; 
}
(*D)[d_node] = node;
node++;
}
} else {
printf("No dangling webpages.\n");
}
int col;
int row;
int idx;
int *elm_count = calloc(node_count, sizeof*elm_count);
for(int edge=0; edge<edge_count; edge++){
col = from_node_id[edge];
row = to_node_id[edge];
elm_count[row]++;
idx = (*row_ptr)[row] + elm_count[row] - 1;
(*col_idx)[idx] = col;
}
int start_idx = 0;
for(int node=0; node<node_count; node++){
sort((*col_idx), start_idx, (*row_ptr)[node+1]);
start_idx = (*row_ptr)[node+1];
}
for(int edge=0; edge<edge_count; edge++){
(*val)[edge] = 1.0/((double)outbound_count[(*col_idx)[edge]]);
}
free(from_node_id);
free(to_node_id);
free(outbound_count);
free(inbound_count);
free(elm_count);
return node_count;
}
void PageRank_iterations(double **val, int **col_idx, int **row_ptr, double **x, double **x_new, int node_count, double damping, double threshold, int **D, int *dangling_count, int threads){
#pragma omp parallel for num_threads(threads)
for(int i=0; i<node_count; i++){
(*x)[i] = 1.0/(double)node_count;
}    
double W;
double temp;
double diff;
int counter_while = 0;
int loop = 1;
while(loop){
W = 0;
diff = 0.0;
for(int d_node=0; d_node<*dangling_count; d_node++){
W += (*x)[(*D)[d_node]];
}
temp = (1 - damping + damping*W)/(double)node_count;
#pragma omp parallel for num_threads(threads)
for(int i=0; i<node_count; i++){
(*x_new)[i] = 0;
for(int j=(*row_ptr)[i]; j<(*row_ptr)[i+1]; j++){
(*x_new)[i] += (*val)[j]*((*x)[(*col_idx)[j]]); 
}
(*x_new)[i] = (*x_new)[i]*damping + temp;
}
for(int i=0; i<node_count; i++){
diff += fabs((*x)[i] - (*x_new)[i]);
(*x)[i] = (*x_new)[i];
}
if(diff < threshold) loop = 0;
counter_while++;
}
printf("Number of PageRank iterations: %d\n", counter_while);
}
void top_n_webpages(double *x, int n, int node_count){
double max = 0;
int rank = 1;
int idx = node_count;
printf("Rank          Page      Score\n");
for(int i=0; i<n; i++){
for(int node=0; node<node_count; node++){
if((x)[node] > max){
max = (x)[node];
idx = node;
}
}
printf("%3d.      %7d       %.10f\n", rank, idx, max);
rank++;
(x)[idx] = 0;
max = 0;
}
}
void sort(int arr[], int beg, int end){
if (end > beg + 1) {
int piv = arr[beg], l = beg + 1, r = end;
while (l < r) {
if (arr[l] <= piv)
l++;
else
swap(&arr[l], &arr[--r]);
}
swap(&arr[--l], &arr[beg]);
sort(arr, beg, l);
sort(arr, r, end);
}
}
void swap(int *a, int *b){
int t=*a; *a=*b; *b=t;
}
