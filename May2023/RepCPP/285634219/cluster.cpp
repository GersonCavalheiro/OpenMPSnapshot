
void invert_cpu(float* data, int actualsize, float* log_determinant)  {
int maxsize = actualsize;
int n = actualsize;
*log_determinant = 0.0;

if (actualsize == 1) { 
*log_determinant = logf(data[0]);
data[0] = 1.0 / data[0];
} else if(actualsize >= 2) { 
for (int i=1; i < actualsize; i++) data[i] /= data[0]; 
for (int i=1; i < actualsize; i++)  { 
for (int j=i; j < actualsize; j++)  { 
float sum = 0.0;
for (int k = 0; k < i; k++)  
sum += data[j*maxsize+k] * data[k*maxsize+i];
data[j*maxsize+i] -= sum;
}
if (i == actualsize-1) continue;
for (int j=i+1; j < actualsize; j++)  {  
float sum = 0.0;
for (int k = 0; k < i; k++)
sum += data[i*maxsize+k]*data[k*maxsize+j];
data[i*maxsize+j] = 
(data[i*maxsize+j]-sum) / data[i*maxsize+i];
}
}

for(int i=0; i<actualsize; i++) {
*log_determinant += ::log10(fabs(data[i*n+i]));
}
for ( int i = 0; i < actualsize; i++ )  
for ( int j = i; j < actualsize; j++ )  {
float x = 1.0;
if ( i != j ) {
x = 0.0;
for ( int k = i; k < j; k++ ) 
x -= data[j*maxsize+k]*data[k*maxsize+i];
}
data[j*maxsize+i] = x / data[j*maxsize+j];
}
for ( int i = 0; i < actualsize; i++ )   
for ( int j = i; j < actualsize; j++ )  {
if ( i == j ) continue;
float sum = 0.0;
for ( int k = i; k < j; k++ )
sum += data[k*maxsize+j]*( (i==k) ? 1.0 : data[i*maxsize+k] );
data[i*maxsize+j] = -sum;
}
for ( int i = 0; i < actualsize; i++ )   
for ( int j = 0; j < actualsize; j++ )  {
float sum = 0.0;
for ( int k = ((i>j)?i:j); k < actualsize; k++ )  
sum += ((j==k)?1.0:data[j*maxsize+k])*data[k*maxsize+i];
data[j*maxsize+i] = sum;
}

} else {
PRINT("Error: Invalid dimensionality for invert(...)\n");
}
}

int validateArguments(int argc, char** argv, int* num_clusters, int* target_num_clusters) {
if(argc <= 5 && argc >= 4) {
if(!sscanf(argv[1],"%d",num_clusters)) {
printf("Invalid number of starting clusters\n\n");
printUsage(argv);
return 1;
} 

if(*num_clusters < 1) {
printf("Invalid number of starting clusters\n\n");
printUsage(argv);
return 1;
}

FILE* infile = fopen(argv[2],"r");
if(!infile) {
printf("Invalid infile.\n\n");
printUsage(argv);
return 2;
} 

if(argc == 5) {
if(!sscanf(argv[4],"%d",target_num_clusters)) {
printf("Invalid number of desired clusters.\n\n");
printUsage(argv);
return 4;
}
if(*target_num_clusters > *num_clusters) {
printf("target_num_clusters must be less than equal to num_clusters\n\n");
printUsage(argv);
return 4;
}
} else {
*target_num_clusters = 0;
}

fclose(infile);
return 0;
} else {
printUsage(argv);
return 1;
}
}

void printUsage(char** argv)
{
printf("Usage: %s num_clusters infile outfile [target_num_clusters]\n",argv[0]);
printf("\t num_clusters: The number of starting clusters\n");
printf("\t infile: ASCII space-delimited FCS data file\n");
printf("\t outfile: Clustering results output file\n");
printf("\t target_num_clusters: A desired number of clusters. Must be less than or equal to num_clusters\n");
}

void writeCluster(FILE* f, clusters_t &clusters, const int c, const int num_dimensions) {
fprintf(f,"Probability: %f\n", clusters.pi[c]);
fprintf(f,"N: %f\n",clusters.N[c]);
fprintf(f,"Means: ");
for(int i=0; i<num_dimensions; i++){
fprintf(f,"%f ",clusters.means[c*num_dimensions+i]);
}
fprintf(f,"\n");

fprintf(f,"\nR Matrix:\n");
for(int i=0; i<num_dimensions; i++) {
for(int j=0; j<num_dimensions; j++) {
fprintf(f,"%f ", clusters.R[c*num_dimensions*num_dimensions+i*num_dimensions+j]);
}
fprintf(f,"\n");
}
fflush(f);   
}


void add_clusters(clusters_t &clusters, const int c1, const int c2, clusters_t &temp_cluster, const int num_dimensions) {
float wt1,wt2;

wt1 = (clusters.N[c1]) / (clusters.N[c1] + clusters.N[c2]);
wt2 = 1.0f - wt1;

for(int i=0; i<num_dimensions;i++) {
temp_cluster.means[i] = wt1*clusters.means[c1*num_dimensions+i] + wt2*clusters.means[c2*num_dimensions+i];
}

for(int i=0; i<num_dimensions; i++) {
for(int j=i; j<num_dimensions; j++) {
temp_cluster.R[i*num_dimensions+j] = ((temp_cluster.means[i]-clusters.means[c1*num_dimensions+i])
*(temp_cluster.means[j]-clusters.means[c1*num_dimensions+j])
+clusters.R[c1*num_dimensions*num_dimensions+i*num_dimensions+j])*wt1;
temp_cluster.R[i*num_dimensions+j] += ((temp_cluster.means[i]-clusters.means[c2*num_dimensions+i])
*(temp_cluster.means[j]-clusters.means[c2*num_dimensions+j])
+clusters.R[c2*num_dimensions*num_dimensions+i*num_dimensions+j])*wt2;
temp_cluster.R[j*num_dimensions+i] = temp_cluster.R[i*num_dimensions+j];
}
}

temp_cluster.pi[0] = clusters.pi[c1] + clusters.pi[c2];

temp_cluster.N[0] = clusters.N[c1] + clusters.N[c2];

float log_determinant;
memcpy(temp_cluster.Rinv,temp_cluster.R,sizeof(float)*num_dimensions*num_dimensions);
invert_cpu(temp_cluster.Rinv,num_dimensions,&log_determinant);
temp_cluster.constant[0] = (-num_dimensions)*0.5f*::logf(2.0f*PI)-0.5f*log_determinant;

temp_cluster.avgvar[0] = clusters.avgvar[0];
}

void copy_cluster(clusters_t &dest, const int c_dest, clusters_t &src, const int c_src, const int num_dimensions) {
dest.N[c_dest] = src.N[c_src];
dest.pi[c_dest] = src.pi[c_src];
dest.constant[c_dest] = src.constant[c_src];
dest.avgvar[c_dest] = src.avgvar[c_src];
memcpy(&(dest.means[c_dest*num_dimensions]),&(src.means[c_src*num_dimensions]),sizeof(float)*num_dimensions);
memcpy(&(dest.R[c_dest*num_dimensions*num_dimensions]),&(src.R[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
memcpy(&(dest.Rinv[c_dest*num_dimensions*num_dimensions]),&(src.Rinv[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
}

void printCluster(clusters_t &clusters, const int c, const int num_dimensions) {
writeCluster(stdout,clusters,c,num_dimensions);
}

float cluster_distance(clusters_t &clusters, const int c1, const int c2, 
clusters_t &temp_cluster, const int num_dimensions) {
add_clusters(clusters,c1,c2,temp_cluster,num_dimensions);

return clusters.N[c1]*clusters.constant[c1] + clusters.N[c2]*clusters.constant[c2] 
- temp_cluster.N[0]*temp_cluster.constant[0];
}


void freeCluster(clusters_t* c) {
free(c->N);
free(c->pi);
free(c->constant);
free(c->avgvar);
free(c->means);
free(c->R);
free(c->Rinv);
free(c->memberships);
}


void setupCluster(clusters_t* c, const int num_clusters, const int num_events, const int num_dimensions) {
c->N = (float*) malloc(sizeof(float)*num_clusters);
c->pi = (float*) malloc(sizeof(float)*num_clusters);
c->constant = (float*) malloc(sizeof(float)*num_clusters);
c->avgvar = (float*) malloc(sizeof(float)*num_clusters);
c->means = (float*) malloc(sizeof(float)*num_dimensions*num_clusters);
c->R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*num_clusters);
c->Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*num_clusters);
c->memberships = (float*) malloc (sizeof(float)*
num_events*(num_clusters+NUM_CLUSTERS_PER_BLOCK-num_clusters % NUM_CLUSTERS_PER_BLOCK));
}


void copyClusterFromDevice(float *N,
float *R,
float *Rinv,
float *pi,
float *constant,
float *avgvar,
float *means,
const int num_clusters, 
const int num_dimensions) {

#pragma omp target update from (N[0:num_clusters])
#pragma omp target update from (R[0:num_dimensions*num_dimensions*num_clusters])
#pragma omp target update from (Rinv[0:num_dimensions*num_dimensions*num_clusters])
#pragma omp target update from (pi[0:num_clusters])
#pragma omp target update from (constant[0:num_clusters])
#pragma omp target update from (avgvar[0:num_clusters])
#pragma omp target update from (means[0:num_dimensions*num_clusters])
}

void copyClusterToDevice(float *N,
float *R,
float *Rinv,
float *pi,
float *constant,
float *avgvar,
float *means,
const int num_clusters, 
const int num_dimensions) {

#pragma omp target update to (N[0:num_clusters])
#pragma omp target update to (R[0:num_dimensions*num_dimensions*num_clusters])
#pragma omp target update to (Rinv[0:num_dimensions*num_dimensions*num_clusters])
#pragma omp target update to (pi[0:num_clusters])
#pragma omp target update to (constant[0:num_clusters])
#pragma omp target update to (avgvar[0:num_clusters])
#pragma omp target update to (means[0:num_dimensions*num_clusters])
}

clusters_t* cluster(int original_num_clusters, int desired_num_clusters, 
int* final_num_clusters, int num_dimensions, int num_events, 
float* fcs_data_by_event) {

int regroup_iterations = 0;
int params_iterations = 0;
int reduce_iterations = 0;
int ideal_num_clusters = original_num_clusters;
int stop_number;

if(desired_num_clusters == 0) {
stop_number = 1;
} else {
stop_number = desired_num_clusters;
}

float* fcs_data_by_dimension  = (float*) malloc(sizeof(float)*num_events*num_dimensions);

for(int e=0; e<num_events; e++) {
for(int d=0; d<num_dimensions; d++) {
if(isnan(fcs_data_by_event[e*num_dimensions+d])) {
printf("Error: Found NaN value in input data. Exiting.\n");
return NULL;
}
fcs_data_by_dimension[d*num_events+e] = fcs_data_by_event[e*num_dimensions+d];
}
}    


PRINT("Number of events: %d\n",num_events);
PRINT("Number of dimensions: %d\n\n",num_dimensions);
PRINT("Starting with %d cluster(s), will stop at %d cluster(s).\n",original_num_clusters,stop_number);

clusters_t clusters;
setupCluster(&clusters, original_num_clusters, num_events, num_dimensions);


clusters_t *saved_clusters = (clusters_t*) malloc(sizeof(clusters_t));
setupCluster(saved_clusters, original_num_clusters, num_events, num_dimensions);

DEBUG("Finished allocating shared cluster structures on host\n");

float likelihood, old_likelihood;
float min_rissanen = FLT_MAX;

clusters_t scratch_cluster;
setupCluster(&scratch_cluster, 1, num_events, num_dimensions);

DEBUG("Finished allocating memory on host for clusters.\n");


float *clusters_N = clusters.N; 
float *clusters_pi = clusters.pi; 
float *clusters_constant = clusters.constant; 
float *clusters_avgvar = clusters.avgvar; 
float *clusters_means = clusters.means; 
float *clusters_R = clusters.R; 
float *clusters_Rinv = clusters.Rinv; 
float *clusters_memberships = clusters.memberships; 
float *likelihoods = (float*) malloc (sizeof(float)*NUM_BLOCKS); 

#pragma omp target data map(alloc:  \
clusters_N[0:original_num_clusters], \
clusters_pi[0:original_num_clusters], \
clusters_constant[0:original_num_clusters], \
clusters_avgvar[0:original_num_clusters], \
clusters_means[0:num_dimensions*original_num_clusters], \
clusters_R[0:num_dimensions*num_dimensions*original_num_clusters], \
clusters_Rinv[0:num_dimensions*num_dimensions*original_num_clusters], \
clusters_memberships[0:num_events*(original_num_clusters+NUM_CLUSTERS_PER_BLOCK-original_num_clusters % NUM_CLUSTERS_PER_BLOCK)],\
likelihoods[0:NUM_BLOCKS]), \
map(to: fcs_data_by_event[0:num_dimensions*num_events], \
fcs_data_by_dimension[0:num_dimensions*num_events])
{
DEBUG("Invoking seed_clusters kernel.\n");

#pragma omp target teams num_teams(1) thread_limit(NUM_THREADS_MSTEP)
{
float means[NUM_DIMENSIONS];
float variances[NUM_DIMENSIONS];
float avgvar; 
float total_variance; 
#pragma omp parallel 
{
int tid = omp_get_thread_num(); 
int num_threads = omp_get_num_threads();
float seed;

int num_elements = num_dimensions*num_dimensions; 


if(tid < num_dimensions) {
means[tid] = 0.0;

for(int i = 0; i < num_events; i++) {
means[tid] += fcs_data_by_event[i*num_dimensions+tid];
}

means[tid] /= (float) num_events;
}

#pragma omp barrier

if(tid < num_dimensions) {
variances[tid] = 0.0;
for(int i = 0; i < num_events; i++) {
variances[tid] += fcs_data_by_event[i*num_dimensions + tid]*
fcs_data_by_event[i*num_dimensions + tid];
}
variances[tid] /= (float) num_events;
variances[tid] -= means[tid]*means[tid];
}

#pragma omp barrier

if(tid == 0) {
total_variance = 0.0;
for(int i=0; i<num_dimensions;i++)
total_variance += variances[i];
avgvar = total_variance / (float) num_dimensions;
}

#pragma omp barrier

if(original_num_clusters > 1) {
seed = (num_events-1.0f)/(original_num_clusters-1.0f);
} else {
seed = 0.0;
}

for(int c=0; c < original_num_clusters; c++) {
if(tid < num_dimensions) {
clusters_means[c*num_dimensions+tid] = fcs_data_by_event[((int)(c*seed))*num_dimensions+tid];
}

for(int i=tid; i < num_elements; i+= num_threads) {
int row = (i) / num_dimensions;
int col = (i) % num_dimensions;

if(row == col) {
clusters_R[c*num_dimensions*num_dimensions+i] = 1.0f;
} else {
clusters_R[c*num_dimensions*num_dimensions+i] = 0.0f;
}
}
if(tid == 0) {
clusters_pi[c] = 1.0f/((float)original_num_clusters);
clusters_N[c] = ((float) num_events) / ((float)original_num_clusters);
clusters_avgvar[c] = avgvar / COVARIANCE_DYNAMIC_RANGE;
}
}
}
}

#pragma omp target teams num_teams(original_num_clusters) thread_limit(NUM_THREADS_MSTEP)
{
float matrix[NUM_DIMENSIONS*NUM_DIMENSIONS];
float determinant_arg;
float sum;
#pragma omp parallel 
{
constants_kernel( clusters_R, 
clusters_Rinv, 
clusters_N, 
clusters_pi, 
clusters_constant, 
clusters_avgvar, 
matrix,
&determinant_arg,
&sum,
original_num_clusters, 
num_dimensions);
}
}

copyClusterFromDevice(clusters_N, clusters_R, clusters_Rinv, clusters_pi, 
clusters_constant, clusters_avgvar, clusters_means, 
original_num_clusters, num_dimensions);

DEBUG("Starting Clusters\n");
for(int c=0; c < original_num_clusters; c++) {
DEBUG("Cluster #%d\n",c);

DEBUG("\tN: %f\n",clusters.N[c]); 
DEBUG("\tpi: %f\n",clusters.pi[c]); 

DEBUG("\tMeans: ");
for(int d=0; d < num_dimensions; d++) {
DEBUG("%.2f ",clusters.means[c*num_dimensions+d]);
}
DEBUG("\n");

DEBUG("\tR:\n\t");
for(int d=0; d < num_dimensions; d++) {
for(int e=0; e < num_dimensions; e++)
DEBUG("%.2f ",clusters.R[c*num_dimensions*num_dimensions+d*num_dimensions+e]);
DEBUG("\n\t");
}
DEBUG("R-inverse:\n\t");
for(int d=0; d < num_dimensions; d++) {
for(int e=0; e < num_dimensions; e++)
DEBUG("%.2f ",clusters.Rinv[c*num_dimensions*num_dimensions+d*num_dimensions+e]);
DEBUG("\n\t");
}
DEBUG("\n");
DEBUG("\tAvgvar: %e\n",clusters.avgvar[c]);
DEBUG("\tConstant: %e\n",clusters.constant[c]);
}

copyClusterToDevice(clusters_N, clusters_R, clusters_Rinv, clusters_pi, 
clusters_constant, clusters_avgvar, clusters_means,
original_num_clusters, num_dimensions);

float epsilon = (1+num_dimensions+0.5f*(num_dimensions+1)*num_dimensions)*
logf((float)num_events*num_dimensions)*0.001f;
int iters;

PRINT("Gaussian.cu: epsilon = %f\n",epsilon);


float distance, min_distance = 0.0;
float rissanen;
int min_c1, min_c2;

for(int num_clusters=original_num_clusters; num_clusters >= stop_number; num_clusters--) {

DEBUG("Invoking E-step kernels.");
#pragma omp target teams num_teams(NUM_BLOCKS*num_clusters) thread_limit(NUM_THREADS_ESTEP)
{
float means[NUM_DIMENSIONS];
float Rinv[NUM_DIMENSIONS*NUM_DIMENSIONS];
#pragma omp parallel 
{
estep1_kernel( 
fcs_data_by_dimension,
clusters_Rinv,
clusters_memberships,
clusters_pi,
clusters_constant,
clusters_means,
means,
Rinv,
num_dimensions, 
num_events); 
}
}

#pragma omp target teams num_teams(NUM_BLOCKS) thread_limit(NUM_THREADS_ESTEP)
{
float total_likelihoods[NUM_THREADS_ESTEP];
#pragma omp parallel 
{
estep2_kernel(
clusters_memberships,
likelihoods,
total_likelihoods,
num_dimensions, 
num_clusters, 
num_events); 
}
}

regroup_iterations++;

#pragma omp target update from (likelihoods[0:NUM_BLOCKS])

likelihood = 0.0;
for(int i=0;i<NUM_BLOCKS;i++) {
likelihood += likelihoods[i]; 
}
DEBUG("Likelihood: %e\n",likelihood);

float change = epsilon*2;

PRINT("Performing EM algorithm on %d clusters.\n",num_clusters);
iters = 0;
while(iters < MIN_ITERS || (fabs(change) > epsilon && iters < MAX_ITERS)) {
old_likelihood = likelihood;

DEBUG("Invoking reestimate_parameters (M-step) kernel.");

#pragma omp target teams num_teams(num_clusters) thread_limit(NUM_THREADS_MSTEP)
{
float temp_sums[NUM_THREADS_MSTEP];
#pragma omp parallel 
{
int tid  = omp_get_thread_num(); 
int num_threads = omp_get_num_threads(); 
int c = omp_get_team_num();

float sum = 0.0f;
for(int event=tid; event < num_events; event += num_threads) {
sum += clusters_memberships[c*num_events+event];
}
temp_sums[tid] = sum;

#pragma omp barrier


for(unsigned int bit = NUM_THREADS_MSTEP >> 1; bit > 0; bit >>= 1) {
float t = temp_sums[tid] + temp_sums[tid^bit];
#pragma omp barrier
temp_sums[tid] = t;
#pragma omp barrier
}
sum = temp_sums[tid];
if(tid == 0) {
clusters_N[c] = sum;
clusters_pi[c] = sum;
}
}
}


#pragma omp target update from (clusters_N[0:num_clusters])


#pragma omp target teams num_teams(num_dimensions*num_clusters) thread_limit(NUM_THREADS_MSTEP)
{
float temp_sum[NUM_THREADS_MSTEP];
#pragma omp parallel 
{
int tid = omp_get_thread_num();
int num_threads = omp_get_num_threads();
int c = omp_get_team_num() / num_dimensions;
int d = omp_get_team_num() % num_dimensions;

float sum = 0.0f;
for(int event=tid; event < num_events; event+= num_threads) {
sum += fcs_data_by_dimension[d*num_events+event]*clusters_memberships[c*num_events+event];
}
temp_sum[tid] = sum;

#pragma omp barrier 

for(unsigned int bit = NUM_THREADS_MSTEP >> 1; bit > 0; bit >>= 1) {
float t = temp_sum[tid] + temp_sum[tid^bit];
#pragma omp barrier
temp_sum[tid] = t;
#pragma omp barrier
}
sum = temp_sum[tid];
if(tid == 0) clusters_means[c*num_dimensions+d] = sum;
}
}

#pragma omp target update from(clusters_means[0:num_clusters*num_dimensions])

for(int c=0; c < num_clusters; c++) {
DEBUG("Cluster %d  Means:", c);
for(int d=0; d < num_dimensions; d++) {
if(clusters.N[c] > 0.5f) {
clusters.means[c*num_dimensions+d] /= clusters.N[c];
} else {
clusters.means[c*num_dimensions+d] = 0.0f;
}
DEBUG(" %f",clusters.means[c*num_dimensions+d]);
}
DEBUG("\n");
}
#pragma omp target update to(clusters_means[0:num_clusters*num_dimensions])



#pragma omp target teams num_teams(num_dimensions*(num_dimensions+1)/2*\
(num_clusters+NUM_CLUSTERS_PER_BLOCK-1)/NUM_CLUSTERS_PER_BLOCK) \
thread_limit(NUM_THREADS_MSTEP)
{
float means_row [NUM_CLUSTERS_PER_BLOCK];
float means_col [NUM_CLUSTERS_PER_BLOCK];
float temp_sums [NUM_THREADS_MSTEP*NUM_CLUSTERS_PER_BLOCK];
#pragma omp parallel 
{
int tid = omp_get_thread_num(); 

int row,col,c1;
compute_row_col(
omp_get_team_num() / ((num_clusters+NUM_CLUSTERS_PER_BLOCK-1)/NUM_CLUSTERS_PER_BLOCK),
num_dimensions, &row, &col);

#pragma omp barrier

c1 = omp_get_team_num() / (num_dimensions*(num_dimensions+1)/2) * NUM_CLUSTERS_PER_BLOCK; 

#if DIAG_ONLY
if(row != col) {
clusters_R[c*num_dimensions*num_dimensions+row*num_dimensions+col] = 0.0f;
clusters_R[c*num_dimensions*num_dimensions+col*num_dimensions+row] = 0.0f;
return;
}
#endif 

if ( (tid < ((num_clusters < NUM_CLUSTERS_PER_BLOCK) ? num_clusters : NUM_CLUSTERS_PER_BLOCK) )  
&& (c1+tid < num_clusters)) { 
means_row[tid] = clusters_means[(c1+tid)*num_dimensions+row];
means_col[tid] = clusters_means[(c1+tid)*num_dimensions+col];
}

#pragma omp barrier


float cov_sum1 = 0.0f;
float cov_sum2 = 0.0f;
float cov_sum3 = 0.0f;
float cov_sum4 = 0.0f;
float cov_sum5 = 0.0f;
float cov_sum6 = 0.0f;
float val1,val2;

for(int c=0; c < NUM_CLUSTERS_PER_BLOCK; c++) {
temp_sums[c*NUM_THREADS_MSTEP+tid] = 0.0;
} 

for(int event=tid; event < num_events; event+=NUM_THREADS_MSTEP) {
val1 = fcs_data_by_dimension[row*num_events+event];
val2 = fcs_data_by_dimension[col*num_events+event];
cov_sum1 += (val1-means_row[0])*(val2-means_col[0])*clusters_memberships[c1*num_events+event]; 
cov_sum2 += (val1-means_row[1])*(val2-means_col[1])*clusters_memberships[(c1+1)*num_events+event]; 
cov_sum3 += (val1-means_row[2])*(val2-means_col[2])*clusters_memberships[(c1+2)*num_events+event]; 
cov_sum4 += (val1-means_row[3])*(val2-means_col[3])*clusters_memberships[(c1+3)*num_events+event]; 
cov_sum5 += (val1-means_row[4])*(val2-means_col[4])*clusters_memberships[(c1+4)*num_events+event]; 
cov_sum6 += (val1-means_row[5])*(val2-means_col[5])*clusters_memberships[(c1+5)*num_events+event]; 
}
temp_sums[0*NUM_THREADS_MSTEP+tid] = cov_sum1;
temp_sums[1*NUM_THREADS_MSTEP+tid] = cov_sum2;
temp_sums[2*NUM_THREADS_MSTEP+tid] = cov_sum3;
temp_sums[3*NUM_THREADS_MSTEP+tid] = cov_sum4;
temp_sums[4*NUM_THREADS_MSTEP+tid] = cov_sum5;
temp_sums[5*NUM_THREADS_MSTEP+tid] = cov_sum6;

#pragma omp barrier

for (int c=0; c < NUM_CLUSTERS_PER_BLOCK; c++) {
float *temp_sum = &temp_sums[c*NUM_THREADS_MSTEP];
for(unsigned int bit = NUM_THREADS_MSTEP >> 1; bit > 0; bit >>= 1) {
float t = temp_sum[tid] + temp_sum[tid^bit];
#pragma omp barrier
temp_sum[tid] = t;
#pragma omp barrier
}
temp_sums[c*NUM_THREADS_MSTEP+tid] = temp_sum[tid];
#pragma omp barrier
}

if (tid == 0) {
for (int c=0; c < NUM_CLUSTERS_PER_BLOCK && (c+c1) < num_clusters; c++) {
int offset = (c+c1)*num_dimensions*num_dimensions;
cov_sum1 = temp_sums[c*NUM_THREADS_MSTEP];
clusters_R[offset+row*num_dimensions+col] = cov_sum1;
clusters_R[offset+col*num_dimensions+row] = cov_sum1;

if(row == col) clusters_R[offset+row*num_dimensions+col] += clusters_avgvar[c+c1];
}
}
}
}

#pragma omp target update from (clusters_R[0:num_clusters*num_dimensions*num_dimensions])

for(int c=0; c < num_clusters; c++) {
if(clusters.N[c] > 0.5f) {
for(int d=0; d < num_dimensions*num_dimensions; d++) {
clusters.R[c*num_dimensions*num_dimensions+d] /= clusters.N[c];
}
} else {
for(int i=0; i < num_dimensions; i++) {
for(int j=0; j < num_dimensions; j++) {
if(i == j) {
clusters.R[c*num_dimensions*num_dimensions+i*num_dimensions+j] = 1.0;
} else {
clusters.R[c*num_dimensions*num_dimensions+i*num_dimensions+j] = 0.0;
}
}
}
}
}

#pragma omp target update to (clusters_R[0:num_clusters*num_dimensions*num_dimensions])


params_iterations++;

DEBUG("Invoking constants kernel.");

#pragma omp target teams num_teams(num_clusters) thread_limit(NUM_THREADS_MSTEP)
{
float matrix[NUM_DIMENSIONS*NUM_DIMENSIONS];
float determinant_arg;
float sum;
#pragma omp parallel 
{
constants_kernel( clusters_R, 
clusters_Rinv, 
clusters_N, 
clusters_pi, 
clusters_constant, 
clusters_avgvar, 
matrix,
&determinant_arg,
&sum,
num_clusters, 
num_dimensions);
}
}


#pragma omp target update from (clusters_constant[0:num_clusters])

for(int temp_c=0; temp_c < num_clusters; temp_c++)
DEBUG("Cluster %d constant: %e\n",temp_c,clusters.constant[temp_c]);

DEBUG("Invoking regroup (E-step) kernel with %d blocks.\n",NUM_BLOCKS);

#pragma omp target teams num_teams(NUM_BLOCKS*num_clusters) thread_limit(NUM_THREADS_ESTEP)
{
float means[NUM_DIMENSIONS];
float Rinv[NUM_DIMENSIONS*NUM_DIMENSIONS];
#pragma omp parallel 
{
estep1_kernel( 
fcs_data_by_dimension,
clusters_Rinv,
clusters_memberships,
clusters_pi,
clusters_constant,
clusters_means,
means,
Rinv,
num_dimensions, 
num_events); 
}
}
#pragma omp target teams num_teams(NUM_BLOCKS) thread_limit(NUM_THREADS_ESTEP)
{
float total_likelihoods[NUM_THREADS_ESTEP];
#pragma omp parallel 
{
estep2_kernel(
clusters_memberships,
likelihoods,
total_likelihoods,
num_dimensions, 
num_clusters, 
num_events); 
}
}

regroup_iterations++;


#pragma omp target update from(likelihoods[0:NUM_BLOCKS])
likelihood = 0.0;
for(int i=0;i<NUM_BLOCKS;i++) likelihood += likelihoods[i]; 

DEBUG("Likelihood: %e\n",likelihood);
change = likelihood - old_likelihood;
DEBUG("GPU 0: Change in likelihood: %e\n",change);
iters++;
}

DEBUG("GPU done with EM loop\n");

copyClusterFromDevice(clusters_N, clusters_R, clusters_Rinv, clusters_pi, 
clusters_constant, clusters_avgvar, clusters_means, 
num_clusters, num_dimensions);

#pragma omp target update from (clusters_memberships[0:num_events*num_clusters])


DEBUG("GPU done with copying cluster data from device\n");

rissanen = -likelihood + 0.5f*(num_clusters*(1.0f+num_dimensions+0.5f*(num_dimensions+1.0f)*num_dimensions)-1.0f)*::logf((float)num_events*num_dimensions);
PRINT("\nLikelihood: %e\n",likelihood);
PRINT("\nRissanen Score: %e\n",rissanen);

if(num_clusters == original_num_clusters || (rissanen < min_rissanen && desired_num_clusters == 0) 
|| (num_clusters == desired_num_clusters)) {
min_rissanen = rissanen;
ideal_num_clusters = num_clusters;
memcpy(saved_clusters->N,clusters.N,sizeof(float)*num_clusters);
memcpy(saved_clusters->pi,clusters.pi,sizeof(float)*num_clusters);
memcpy(saved_clusters->constant,clusters.constant,sizeof(float)*num_clusters);
memcpy(saved_clusters->avgvar,clusters.avgvar,sizeof(float)*num_clusters);
memcpy(saved_clusters->means,clusters.means,sizeof(float)*num_dimensions*num_clusters);
memcpy(saved_clusters->R,clusters.R,sizeof(float)*num_dimensions*num_dimensions*num_clusters);
memcpy(saved_clusters->Rinv,clusters.Rinv,sizeof(float)*num_dimensions*num_dimensions*num_clusters);
memcpy(saved_clusters->memberships,clusters.memberships,sizeof(float)*num_events*num_clusters);
}

if(num_clusters > stop_number) {
for(int i=num_clusters-1; i >= 0; i--) {
if(clusters.N[i] < 0.5) {
DEBUG("Cluster #%d has less than 1 data point in it.\n",i);
for(int j=i; j < num_clusters-1; j++) {
copy_cluster(clusters,j,clusters,j+1,num_dimensions);
}
num_clusters--;
}
}

min_c1 = 0;
min_c2 = 1;
DEBUG("Number of non-empty clusters: %d\n",num_clusters); 
for(int c1=0; c1<num_clusters;c1++) {
for(int c2=c1+1; c2<num_clusters;c2++) {
distance = cluster_distance(clusters,c1,c2,scratch_cluster,num_dimensions);
if((c1 ==0 && c2 == 1) || distance < min_distance) {
min_distance = distance;
min_c1 = c1;
min_c2 = c2;
}
}
}

PRINT("\nMinimum distance between (%d,%d). Combining clusters\n",min_c1,min_c2);
add_clusters(clusters,min_c1,min_c2,scratch_cluster,num_dimensions);

copy_cluster(clusters,min_c1,scratch_cluster,0,num_dimensions);

for(int i=min_c2; i < num_clusters-1; i++) {
copy_cluster(clusters,i,clusters,i+1,num_dimensions);
}

copyClusterToDevice(clusters_N, clusters_R, clusters_Rinv, clusters_pi, 
clusters_constant, clusters_avgvar, clusters_means,
num_clusters, num_dimensions);

} 
reduce_iterations++;

} 
PRINT("\nFinal rissanen Score was: %f, with %d clusters.\n",min_rissanen,ideal_num_clusters);

}

freeCluster(&scratch_cluster);
freeCluster(&clusters);
free(fcs_data_by_dimension);
free(likelihoods);

*final_num_clusters = ideal_num_clusters;
return saved_clusters;
}
