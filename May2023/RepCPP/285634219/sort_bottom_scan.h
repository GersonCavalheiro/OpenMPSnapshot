int group_range = omp_get_num_teams();
int group = omp_get_team_num();
int local_range = omp_get_num_threads();
int lid = omp_get_thread_num();



int histogram[16];


int n4 = size / 4; 

int region_size = n4 / group_range;
int block_start = group * region_size;
int block_stop  = (group == group_range - 1) ?  n4 : block_start + region_size;

int i = block_start + lid;
int window = block_start;

if (lid < 16)
{
l_block_counts[lid] = 0;
l_scanned_seeds[lid] = isums[(lid*group_range)+group];
}
#pragma omp barrier

while (window < block_stop)
{
for (int q = 0; q < 16; q++) histogram[q] = 0;
vec4<T> val_4;
vec4<T> key_4;

if (i < block_stop) 
{
val_4.x = in[4*i];
val_4.y = in[4*i+1];
val_4.z = in[4*i+2];
val_4.w = in[4*i+3];

key_4.x = (val_4.x >> shift) & 0xFU;
key_4.y = (val_4.y >> shift) & 0xFU;
key_4.z = (val_4.z >> shift) & 0xFU;
key_4.w = (val_4.w >> shift) & 0xFU;

histogram[key_4.x]++;
histogram[key_4.y]++;
histogram[key_4.z]++;
histogram[key_4.w]++;
}

for (int digit = 0; digit < 16; digit++)
{
int idx = lid;
lmem[idx] = 0;
idx += local_range;
lmem[idx] = histogram[digit];
#pragma omp barrier
for (int i = 1; i < local_range; i *= 2)
{
T t = lmem[idx -  i]; 
#pragma omp barrier
lmem[idx] += t;     
#pragma omp barrier
}
histogram[digit] = lmem[idx-1];

#pragma omp barrier
}

if (i < block_stop) 
{
int address;
address = histogram[key_4.x] + l_scanned_seeds[key_4.x] + l_block_counts[key_4.x];
out[address] = val_4.x;
histogram[key_4.x]++;

address = histogram[key_4.y] + l_scanned_seeds[key_4.y] + l_block_counts[key_4.y];
out[address] = val_4.y;
histogram[key_4.y]++;

address = histogram[key_4.z] + l_scanned_seeds[key_4.z] + l_block_counts[key_4.z];
out[address] = val_4.z;
histogram[key_4.z]++;

address = histogram[key_4.w] + l_scanned_seeds[key_4.w] + l_block_counts[key_4.w];
out[address] = val_4.w;
histogram[key_4.w]++;
}

#pragma omp barrier
if (lid == local_range-1)
{
for (int q = 0; q < 16; q++)
{
l_block_counts[q] += histogram[q];
}
}
#pragma omp barrier

window += local_range;
i += local_range;
}

