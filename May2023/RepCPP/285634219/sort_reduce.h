
int group_range = omp_get_num_teams();
int group = omp_get_team_num();
int local_range = omp_get_num_threads();

int region_size = ((size / 4) / group_range) * 4;
int block_start = group * region_size;

int block_stop  = (group == group_range - 1) ?  size : block_start + region_size;

int tid = omp_get_thread_num();
int i = block_start + tid;

int digit_counts[16] = { 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0 };

while (i < block_stop) 
{

digit_counts[(in[i] >> shift) & 0xFU]++;
i += local_range;
}

for (int d = 0; d < 16; d++)
{
lmem[tid] = digit_counts[d];
#pragma omp barrier

for (unsigned int s = local_range / 2; s > 0; s >>= 1)
{
if (tid < s)
{
lmem[tid] += lmem[tid + s];
}
#pragma omp barrier
}

if (tid == 0)
{
isums[(d * group_range) + group] = lmem[0];
}
}
