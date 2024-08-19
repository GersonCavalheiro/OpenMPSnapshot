
#pragma omp target teams distribute parallel for thread_limit(work_group_size)
for (int gid = 0; gid < global_work_size; gid++) {

int i = gid/local_work_size + MAX_RAD + 2;
int j = gid%local_work_size + MAX_RAD + 2;

float max_GICOV = 0.f;

for (int k = 0; k < NCIRCLES; k++) {
float sum = 0.f, M2 = 0.f, mean = 0.f;    

for (int n = 0; n < NPOINTS; n++) {
int y = j + host_tY[(k * NPOINTS) + n];
int x = i + host_tX[(k * NPOINTS) + n];

int addr = x * grad_m + y;
float p = host_grad_x[addr] * host_cos_angle[n] + 
host_grad_y[addr] * host_sin_angle[n];

sum += p;

float delta = p - mean;
mean = mean + (delta / (float) (n + 1));
M2 = M2 + (delta * (p - mean));
}

mean = sum / ((float) NPOINTS);

float var = M2 / ((float) (NPOINTS - 1));

if (((mean * mean) / var) > max_GICOV) max_GICOV = (mean * mean) / var;
}

host_gicov[(i * grad_m) + j] = max_GICOV;
}
