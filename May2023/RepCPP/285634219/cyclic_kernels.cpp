


void cyclic_small_systems_kernel(
const float* a_d, 
const float* b_d, 
const float* c_d, 
const float* d_d, 
float* x_d, 
const int system_size, 
const int num_systems, 
const int iterations,
const size_t szTeams,
const size_t szThreads)
{
#pragma omp target teams num_teams(szTeams) thread_limit(szThreads)
{
float shared[(SYSTEM_SIZE+1)*5];
#pragma omp parallel 
{
int thid = omp_get_thread_num();
int blid = omp_get_team_num();

int stride = 1;
int half_size = system_size >> 1;
int thid_num = half_size;

float* a = shared;
float* b = &a[system_size];
float* c = &b[system_size];
float* d = &c[system_size];
float* x = &d[system_size];

a[thid] = a_d[thid + blid * system_size];
a[thid + thid_num] = a_d[thid + thid_num + blid * system_size];

b[thid] = b_d[thid + blid * system_size];
b[thid + thid_num] = b_d[thid + thid_num + blid * system_size];

c[thid] = c_d[thid + blid * system_size];
c[thid + thid_num] = c_d[thid + thid_num + blid * system_size];

d[thid] = d_d[thid + blid * system_size];
d[thid + thid_num] = d_d[thid + thid_num + blid * system_size];

#pragma omp barrier

for (int j = 0; j < iterations; j++)
{
#pragma omp barrier

stride <<= 1;
int delta = stride >> 1;
if (thid < thid_num)
{ 
int i = stride * thid + stride - 1;

if (i == system_size - 1)
{
float tmp = a[i] / b[i-delta];
b[i] = b[i] - c[i-delta] * tmp;
d[i] = d[i] - d[i-delta] * tmp;
a[i] = -a[i-delta] * tmp;
c[i] = 0;      
}
else
{
float tmp1 = a[i] / b[i-delta];
float tmp2 = c[i] / b[i+delta];
b[i] = b[i] - c[i-delta] * tmp1 - a[i+delta] * tmp2;
d[i] = d[i] - d[i-delta] * tmp1 - d[i+delta] * tmp2;
a[i] = -a[i-delta] * tmp1;
c[i] = -c[i+delta] * tmp2;
}
}
thid_num >>= 1;
}

if (thid < 2)
{
int addr1 = stride - 1;
int addr2 = (stride << 1) - 1;
float tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
}

thid_num = 2;
for (int j = 0; j < iterations; j++)
{
int delta = stride >> 1;
#pragma omp barrier
if (thid < thid_num)
{
int i = stride * thid + (stride >> 1) - 1;
if (i == delta - 1)
x[i] = (d[i] - c[i] * x[i+delta]) / b[i];
else
x[i] = (d[i] - a[i] * x[i-delta] - c[i] * x[i+delta]) / b[i];
}
stride >>= 1;
thid_num <<= 1;
}

#pragma omp barrier   

x_d[thid + blid * system_size] = x[thid];
x_d[thid + half_size + blid * system_size] = x[thid + half_size];
}
}
}

void cyclic_branch_free_kernel(
const float* a_d, 
const float* b_d, 
const float* c_d, 
const float* d_d, 
float* x_d, 
const int system_size, 
const int num_systems, 
const int iterations,
const size_t szTeams,
const size_t szThreads)
{
#pragma omp target teams num_teams(szTeams) thread_limit(szThreads)
{
float shared[(SYSTEM_SIZE+1)*5];
#pragma omp parallel 
{

int thid = omp_get_thread_num();
int blid = omp_get_team_num();

int stride = 1;
int half_size = system_size >> 1;
int thid_num = half_size;

float* a = shared;
float* b = &a[system_size];
float* c = &b[system_size];
float* d = &c[system_size];
float* x = &d[system_size];

a[thid] = a_d[thid + blid * system_size];
a[thid + thid_num] = a_d[thid + thid_num + blid * system_size];

b[thid] = b_d[thid + blid * system_size];
b[thid + thid_num] = b_d[thid + thid_num + blid * system_size];

c[thid] = c_d[thid + blid * system_size];
c[thid + thid_num] = c_d[thid + thid_num + blid * system_size];

d[thid] = d_d[thid + blid * system_size];
d[thid + thid_num] = d_d[thid + thid_num + blid * system_size];

#pragma omp barrier

for (int j = 0; j < iterations; j++)
{
#pragma omp barrier

stride <<= 1;
int delta = stride >> 1;
if (thid < thid_num)
{ 
int i = stride * thid + stride - 1;
int iRight = i+delta;
iRight = iRight & (system_size-1);
float tmp1 = a[i] / b[i-delta];
float tmp2 = c[i] / b[iRight];
b[i] = b[i] - c[i-delta] * tmp1 - a[iRight] * tmp2;
d[i] = d[i] - d[i-delta] * tmp1 - d[iRight] * tmp2;
a[i] = -a[i-delta] * tmp1;
c[i] = -c[iRight]  * tmp2;
}

thid_num >>= 1;
}

if (thid < 2)
{
int addr1 = stride - 1;
int addr2 = (stride << 1) - 1;
float tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
}

thid_num = 2;
for (int j = 0; j < iterations; j++)
{
int delta = stride >> 1;
#pragma omp barrier
if (thid < thid_num)
{
int i = stride * thid + (stride >> 1) - 1;
if (i == delta - 1)
x[i] = (d[i] - c[i] * x[i+delta]) / b[i];
else
x[i] = (d[i] - a[i] * x[i-delta] - c[i] * x[i+delta]) / b[i];
}
stride >>= 1;
thid_num <<= 1;
}

#pragma omp barrier   

x_d[thid + blid * system_size] = x[thid];
x_d[thid + half_size + blid * system_size] = x[thid + half_size];
}
}
}

