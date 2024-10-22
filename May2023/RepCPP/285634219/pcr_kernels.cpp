




void pcr_small_systems_kernel(
const float* a_d, 
const float* b_d, 
const float* c_d, 
const float* d_d, 
float* x_d, 
int system_size,
int num_systems, 
int iterations,
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

int delta = 1;

float* a = shared;
float* b = &a[system_size+1];
float* c = &b[system_size+1];
float* d = &c[system_size+1];
float* x = &d[system_size+1];

a[thid] = a_d[thid + blid * system_size];
b[thid] = b_d[thid + blid * system_size];
c[thid] = c_d[thid + blid * system_size];
d[thid] = d_d[thid + blid * system_size];

float aNew, bNew, cNew, dNew;

#pragma omp barrier

for (int j = 0; j < iterations; j++)
{
int i = thid;

if(i < delta)
{
float tmp2 = c[i] / b[i+delta];
bNew = b[i] - a[i+delta] * tmp2;
dNew = d[i] - d[i+delta] * tmp2;
aNew = 0;
cNew = -c[i+delta] * tmp2;  
}
else if((system_size-i-1) < delta)
{
float tmp = a[i] / b[i-delta];
bNew = b[i] - c[i-delta] * tmp;
dNew = d[i] - d[i-delta] * tmp;
aNew = -a[i-delta] * tmp;
cNew = 0;      
}
else        
{
float tmp1 = a[i] / b[i-delta];
float tmp2 = c[i] / b[i+delta];
bNew = b[i] - c[i-delta] * tmp1 - a[i+delta] * tmp2;
dNew = d[i] - d[i-delta] * tmp1 - d[i+delta] * tmp2;
aNew = -a[i-delta] * tmp1;
cNew = -c[i+delta] * tmp2;
}

#pragma omp barrier

b[i] = bNew;
d[i] = dNew;
a[i] = aNew;
c[i] = cNew;  

delta *= 2;
#pragma omp barrier
}

if (thid < delta)
{
int addr1 = thid;
int addr2 = thid + delta;
float tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
}

#pragma omp barrier

x_d[thid + blid * system_size] = x[thid];
}
}
}

void pcr_branch_free_kernel(
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

int delta = 1;

float* a = shared;
float* b = &a[system_size+1];
float* c = &b[system_size+1];
float* d = &c[system_size+1];
float* x = &d[system_size+1];

a[thid] = a_d[thid + blid * system_size];
b[thid] = b_d[thid + blid * system_size];
c[thid] = c_d[thid + blid * system_size];
d[thid] = d_d[thid + blid * system_size];

float aNew, bNew, cNew, dNew;

#pragma omp barrier

for (int j = 0; j < iterations; j++)
{
int i = thid;

int iRight = i+delta;
iRight = iRight & (system_size-1);

int iLeft = i-delta;
iLeft = iLeft & (system_size-1);

float tmp1 = a[i] / b[iLeft];
float tmp2 = c[i] / b[iRight];

bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
dNew = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
aNew = -a[iLeft] * tmp1;
cNew = -c[iRight] * tmp2;

#pragma omp barrier

b[i] = bNew;
d[i] = dNew;
a[i] = aNew;
c[i] = cNew;  

delta *= 2;
#pragma omp barrier
}

if (thid < delta)
{
int addr1 = thid;
int addr2 = thid + delta;
float tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
}

#pragma omp barrier
x_d[thid + blid * system_size] = x[thid];
}
}
}

