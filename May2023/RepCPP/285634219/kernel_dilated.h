#pragma omp target teams distribute parallel for thread_limit(local_work_size)
for (int thread_id = 0; thread_id < global_work_size; thread_id++)  
{
int el_center_i = strel_m / 2;
int el_center_j = strel_n / 2;

int i = thread_id % max_gicov_m;
int j = thread_id / max_gicov_m;

float max = 0.0f;

int el_i, el_j, x, y;

for (el_i = 0; el_i < strel_m; el_i++) {
y = i - el_center_i + el_i;
if ( (y >= 0) && (y < max_gicov_m) ) {
for (el_j = 0; el_j < strel_n; el_j++) {
x = j - el_center_j + el_j;
if ( (x >= 0) && (x < max_gicov_n) &&
(host_strel[(el_i * strel_n) + el_j] != 0) ) {
int addr = (x * max_gicov_m) + y;
float temp = host_gicov[addr];
if (temp > max) max = temp;
}
}
}
}

host_dilated[(i * max_gicov_n) + j] = max;
}
