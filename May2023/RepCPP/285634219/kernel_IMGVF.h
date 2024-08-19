#pragma omp target teams num_teams(num_cells) thread_limit(LOCAL_WORK_SIZE)
{
float IMGVF[41*81];
float IMGVF_buffer[LOCAL_WORK_SIZE];
int cell_converged;
#pragma omp parallel
{
int cell_num = omp_get_team_num();

int I_offset = host_I_offsets[cell_num];
float* IMGVF_global = &(host_IMGVF_all[I_offset]);

int m = host_m_array[cell_num];
int n = host_n_array[cell_num];

int IMGVF_Size = m * n;
int tb_count = (m * n + LOCAL_WORK_SIZE - 1) / LOCAL_WORK_SIZE;

int thread_id = omp_get_thread_num();
int thread_block, i, j;
for (thread_block = 0; thread_block < tb_count; thread_block++) {
int offset = thread_block * LOCAL_WORK_SIZE;
i = (thread_id + offset) / n;
j = (thread_id + offset) % n;
if (i < m) IMGVF[(i * n) + j] = IMGVF_global[(i * n) + j];
}
#pragma omp barrier

if (thread_id == 0) cell_converged = 0;
#pragma omp barrier

const float one_nth = 1.0f / (float) n;
const int tid_mod = thread_id % n;
const int tbsize_mod = LOCAL_WORK_SIZE % n;

FP_TYPE one_over_e = FP_CONST(1.0) / e_float;

int iterations = 0;
while ((! cell_converged) && (iterations < max_iterations)) {

FP_TYPE total_diff = FP_CONST(0.0);

int old_i = 0, old_j = 0;
j = tid_mod - tbsize_mod;

for (thread_block = 0; thread_block < tb_count; thread_block++) {
old_i = i;
old_j = j;

int offset = thread_block * LOCAL_WORK_SIZE;
i = (thread_id + offset) * one_nth;
j += tbsize_mod;
if (j >= n) j -= n;

FP_TYPE new_val = FP_CONST(0.0);
FP_TYPE old_val = FP_CONST(0.0);

if (i < m) {
int rowU = (i == 0) ? 0 : i - 1;
int rowD = (i == m - 1) ? m - 1 : i + 1;
int colL = (j == 0) ? 0 : j - 1;
int colR = (j == n - 1) ? n - 1 : j + 1;

old_val    = IMGVF[(i * n) + j];
FP_TYPE U  = IMGVF[(rowU * n) + j   ] - old_val;
FP_TYPE D  = IMGVF[(rowD * n) + j   ] - old_val;
FP_TYPE L  = IMGVF[(i    * n) + colL] - old_val;
FP_TYPE R  = IMGVF[(i    * n) + colR] - old_val;
FP_TYPE UR = IMGVF[(rowU * n) + colR] - old_val;
FP_TYPE DR = IMGVF[(rowD * n) + colR] - old_val;
FP_TYPE UL = IMGVF[(rowU * n) + colL] - old_val;
FP_TYPE DL = IMGVF[(rowD * n) + colL] - old_val;

FP_TYPE UHe  = heaviside((U  *       -vy_float)  * one_over_e);
FP_TYPE DHe  = heaviside((D  *        vy_float)  * one_over_e);
FP_TYPE LHe  = heaviside((L  *  -vx_float     )  * one_over_e);
FP_TYPE RHe  = heaviside((R  *   vx_float     )  * one_over_e);
FP_TYPE URHe = heaviside((UR * ( vx_float - vy_float)) * one_over_e);
FP_TYPE DRHe = heaviside((DR * ( vx_float + vy_float)) * one_over_e);
FP_TYPE ULHe = heaviside((UL * (-vx_float - vy_float)) * one_over_e);
FP_TYPE DLHe = heaviside((DL * (-vx_float + vy_float)) * one_over_e);

new_val = old_val + (MU / LAMBDA) * (UHe  * U  + DHe  * D  + LHe  * L  + RHe  * R +
URHe * UR + DRHe * DR + ULHe * UL + DLHe * DL);
FP_TYPE vI = host_I_all[I_offset+ (i * n) + j];
new_val -= ((1.0 / LAMBDA) * vI * (new_val - vI));

}
if (thread_block > 0) {
offset = (thread_block - 1) * LOCAL_WORK_SIZE;
if (old_i < m) IMGVF[(old_i * n) + old_j] = IMGVF_buffer[thread_id];
}
if (thread_block < tb_count - 1) {
IMGVF_buffer[thread_id] = new_val;
} else {
if (i < m) IMGVF[(i * n) + j] = new_val;
}

total_diff += fabsf(new_val - old_val);

#pragma omp barrier
}

IMGVF_buffer[thread_id] = total_diff;
#pragma omp barrier

if (thread_id >= NEXT_LOWEST_POWER_OF_TWO) {
IMGVF_buffer[thread_id - NEXT_LOWEST_POWER_OF_TWO] += IMGVF_buffer[thread_id];
}
#pragma omp barrier

int th;
for (th = NEXT_LOWEST_POWER_OF_TWO / 2; th > 0; th /= 2) {
if (thread_id < th) {
IMGVF_buffer[thread_id] += IMGVF_buffer[thread_id + th];
}
#pragma omp barrier
}

if(thread_id == 0) {
FP_TYPE mean = IMGVF_buffer[thread_id] / (FP_TYPE) (m * n);
if (mean < cutoff_float) {
cell_converged = 1;
}
}

#pragma omp barrier

iterations++;
}

for (thread_block = 0; thread_block < tb_count; thread_block++) {
int offset = thread_block * LOCAL_WORK_SIZE + thread_id;
if (offset < IMGVF_Size)
IMGVF_global[offset] = IMGVF[offset];
}
}
}
