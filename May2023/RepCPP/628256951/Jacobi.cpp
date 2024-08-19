class Jacobi {
public:

Jacobi() {}


int jacobi_serial(double *X,
double *B, 
double *A, 
double TOL, 
int max_iterations,  
int N) {

int iteration = 0;
for(iteration=0; iteration<max_iterations; iteration++) {

double *old_X = copy_vector(X, N);
double local_errors[N];
for(int i=0; i<N; i++) {





if(i > 0 && i < N-1) {
X[i] = (1.0f / A[i * N + i]) * (B[i] - (A[i * N + (i-1)]*old_X[i-1]) - (A[i * N + (i+1)]*old_X[i+1]));
} 
else if(i == N-1) {
X[i] = (1.0f / A[i * N + i]) * (B[i] - (A[i * N + (i-1)]*old_X[i-1]));
} 
else {
X[i] = (1.0f / A[i * N + i]) * (B[i] - (A[i * N + (i+1)]*old_X[i+1]));
} 

local_errors[i] = fabs(X[i] - old_X[i]); 
}
free(old_X);
double *max_error = std::max_element(local_errors, local_errors+N);

if(*max_error < TOL) 
break;

}

return iteration;
}


int jacobi_parallel(double *X, 
double *B, 
double *A, 
double TOL,
int max_iterations, 
const std::vector<std::vector<int>>& partitions, 
int N, 
const int THREAD_COUNT) {

int iteration = 0;
for(iteration=0; iteration<max_iterations; iteration++) {

double *old_X = copy_vector(X, N);
double local_errors[N];
#pragma omp parallel num_threads(THREAD_COUNT) shared(local_errors, X, A, B)
{
#pragma omp for
for(int p=0; p<partitions.size(); p++) {

for(int variable: partitions[p]) {








if(variable > 0 && variable < N-1) {
X[variable] = (1.0f / A[variable * N + variable]) * (B[variable] - (A[variable * N + (variable-1)]*old_X[variable-1]) - (A[variable * N + (variable+1)]*old_X[variable+1]));
} 
else if(variable == N-1) {
X[variable] = (1.0f / A[variable * N + variable]) * (B[variable] - (A[variable * N + (variable-1)]*old_X[variable-1]));
} 
else {
X[variable] = (1.0f / A[variable * N + variable]) * (B[variable] - (A[variable * N + (variable+1)]*old_X[variable+1]));
} 
local_errors[variable] = fabs(X[variable] - old_X[variable]);                  
}

}
#pragma omp barrier
}
free(old_X);
double *max_error = std::max_element(local_errors, local_errors+N);

if(*max_error < TOL) {
break;
}  
}
return iteration;
}




};
