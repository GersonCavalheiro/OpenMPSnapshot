


void sweep_small_systems_local_kernel(
const float* a_d, 
const float* b_d, 
const float* c_d, 
const float* d_d, 
float* x_d, 
const int system_size, 
const int num_systems,
const bool reorder,
const size_t szTeams,
const size_t szThreads)
{
#pragma omp target teams distribute parallel for num_teams(szTeams) thread_limit(szThreads)
for (int i = 0; i < num_systems; i++) {
int stride = reorder ? num_systems: 1;
int base_idx = reorder ? i : i * system_size;

float a[128];

float c1, c2, c3;
float f_i, x_prev, x_next;


c1 = c_d[base_idx];
c2 = b_d[base_idx];
f_i = d_d[base_idx];

a[1] = - c1 / c2;
x_prev = f_i / c2;

int idx = base_idx;
x_d[base_idx] = x_prev;
for (int k = 1; k < system_size-1; k++)
{
idx += stride;

c1 = c_d[idx];
c2 = b_d[idx];
c3 = a_d[idx];
f_i = d_d[idx];

float q = (c3 * a[k] + c2);
float t = 1 / q; 
x_next = (f_i - c3 * x_prev) * t;
x_d[idx] = x_prev = x_next;

a[k+1] = - c1 * t;
}

idx += stride;

c2 = b_d[idx];
c3 = a_d[idx];
f_i = d_d[idx];

float q = (c3 * a[system_size-1] + c2);
float t = 1 / q; 
x_next = (f_i - c3 * x_prev) * t;
x_d[idx] = x_prev = x_next;

for (int k = system_size-2; k >= 0; k--)
{
idx -= stride;
x_next = x_d[idx];
x_next += x_prev * a[k+1];
x_d[idx] = x_prev = x_next;
}
}
}

#pragma omp declare target
inline int getLocalIdx(int i, int k, int num_systems)
{
return i + num_systems * k;

}
#pragma omp end declare target

void sweep_small_systems_global_kernel(
const float* a_d, 
const float* b_d, 
const float* c_d, 
const float* d_d, 
float* x_d,
float* w_d,
const int system_size, 
const int num_systems,
const bool reorder,
const size_t szTeams,
const size_t szThreads)
{
#pragma omp target teams distribute parallel for num_teams(szTeams) thread_limit(szThreads)
for (int i = 0; i < num_systems; i++) {

int stride = reorder ? num_systems: 1;
int base_idx = reorder ? i : i * system_size;

float c1, c2, c3;
float f_i, x_prev, x_next;


c1 = c_d[base_idx];
c2 = b_d[base_idx];
f_i = d_d[base_idx];

w_d[getLocalIdx(i, 1, num_systems)] = - c1 / c2;
x_prev = f_i / c2;

int idx = base_idx;
x_d[base_idx] = x_prev;
for (int k = 1; k < system_size-1; k++)
{
idx += stride;

c1 = c_d[idx];
c2 = b_d[idx];
c3 = a_d[idx];
f_i = d_d[idx];

float q = (c3 * w_d[getLocalIdx(i, k, num_systems)] + c2);
float t = 1 / q; 
x_next = (f_i - c3 * x_prev) * t;
x_d[idx] = x_prev = x_next;

w_d[getLocalIdx(i, k+1, num_systems)] = - c1 * t;
}

idx += stride;

c2 = b_d[idx];
c3 = a_d[idx];
f_i = d_d[idx];

float q = (c3 * w_d[getLocalIdx(i, system_size-1, num_systems)] + c2);
float t = 1 / q; 
x_next = (f_i - c3 * x_prev) * t;
x_d[idx] = x_prev = x_next;

for (int k = system_size-2; k >= 0; k--)
{
idx -= stride;
x_next = x_d[idx];
x_next += x_prev * w_d[getLocalIdx(i, k+1, num_systems)];
x_d[idx] = x_prev = x_next;
}
}
}

inline float4 load(const float* a, int i)
{
return {a[i], a[i+1], a[i+2], a[i+3]};
}

inline void store(float* a, int i, float4 v)
{
a[i] = v.x;
a[i+1] = v.y;
a[i+2] = v.z;
a[i+3] = v.w;
}

inline float4 operator*(float4 a, float4 b)
{
return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

inline float4 operator/(float4 a, float4 b)
{
return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}

inline float4 operator+(float4 a, float4 b)
{
return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

inline float4 operator-(float4 a, float4 b)
{
return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

inline float4 operator-(float4 &a)
{
return {-a.x, -a.y, -a.z, -a.w};
}

inline void operator+=(float4 &a, float4 b)
{
a.x += b.x;
a.y += b.y;
a.z += b.z;
a.w += b.w;
}

void sweep_small_systems_global_vec4_kernel(
const float* a_d, 
const float* b_d, 
const float* c_d, 
const float* d_d, 
float* x_d, 
float* w_d, 
const int system_size, 
const int num_systems,
const bool reorder,
const size_t szTeams,
const size_t szThreads)
{
#pragma omp target teams distribute parallel for num_teams(szTeams) thread_limit(szThreads)
for (int j = 0 ; j < num_systems; j++) {

int i = j << 2;

if (i < num_systems) {

int stride = reorder ? num_systems: 4;
int base_idx = reorder ? i : i * system_size;

float4 c1, c2, c3;
float4 f_i, x_prev, x_next;


c1 = load(c_d, base_idx);
c2 = load(b_d, base_idx);
f_i = load(d_d, base_idx);

store(w_d, getLocalIdx(i, 1, num_systems), - c1 / c2);
x_prev = f_i / c2;

int idx = base_idx;
store(x_d, base_idx, x_prev);
for (int k = 1; k < system_size-1; k++)
{
idx += stride;

c1 = load(c_d, idx);
c2 = load(b_d, idx);
c3 = load(a_d, idx);
f_i = load(d_d, idx);

float4 q = (c3 * load(w_d, getLocalIdx(i, k, num_systems)) + c2);
float4 t = {1.0f/q.x, 1.0f/q.y, 1.0f/q.z, 1.0f/q.w};
x_next = (f_i - c3 * x_prev) * t;
x_prev = x_next;
store(x_d, idx, x_prev);

store(w_d, getLocalIdx(i, k+1, num_systems), - c1 * t);
}

idx += stride;

c2 = load(b_d, idx);
c3 = load(a_d, idx);
f_i = load(d_d, idx);

float4 q = (c3 * load(w_d, getLocalIdx(i, system_size-1, num_systems)) + c2);
float4 t = {1.0f/q.x, 1.0f/q.y, 1.0f/q.z, 1.0f/q.w};
x_next = (f_i - c3 * x_prev) * t;
x_prev = x_next;
store(x_d, idx, x_prev);

for (int k = system_size-2; k >= 0; k--)
{
idx -= stride;
x_next = load(x_d, idx);
x_next += x_prev * load(w_d, getLocalIdx(i, k+1, num_systems));
x_prev = x_next;
store(x_d, idx, x_prev); 
}
}
}
}

void transpose(
float* odata, 
const float* idata, 
const int width, 
const int height,
const size_t szTeamX,
const size_t szTeam) 
{
#pragma omp target teams num_teams(szTeam) thread_limit(TRANSPOSE_BLOCK_DIM * TRANSPOSE_BLOCK_DIM)
{
float block[TRANSPOSE_BLOCK_DIM * (TRANSPOSE_BLOCK_DIM+1)];
#pragma omp parallel 
{
int blockIdxx = omp_get_team_num() % szTeamX;
int blockIdxy = omp_get_team_num() / szTeamX;

int threadIdxx = omp_get_thread_num() % TRANSPOSE_BLOCK_DIM;
int threadIdxy = omp_get_thread_num() / TRANSPOSE_BLOCK_DIM; 

int i0 = (blockIdxx * BLOCK_DIM) + threadIdxx;
int j0 = (blockIdxy * BLOCK_DIM) + threadIdxy;
int i1, j1, idx_a, idx_b;

if (i0 < width && j0 < height) {

i1 = (blockIdxy * BLOCK_DIM) + threadIdxx;
j1 = (blockIdxx * BLOCK_DIM) + threadIdxy;

if (i1 < height && j1 < width) {

idx_a = i0 + (j0 * width);
idx_b = i1 + (j1 * height);

block[threadIdxy * (BLOCK_DIM+1) + threadIdxx] = idata[idx_a];
}
}

#pragma omp barrier

if (i0 < width && j0 < height && i1 < height && j1 < width) 
odata[idx_b] = block[threadIdxx * (BLOCK_DIM+1) + threadIdxy];
}
}
}
