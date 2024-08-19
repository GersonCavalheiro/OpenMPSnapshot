


template <
int         NUM_PARTS,
int         ACTIVE_CHANNELS,
int         NUM_BINS,
typename    PixelType>
__global__ void histogram_gmem_atomics(
const PixelType *in,
int width,
int height,
unsigned int *out)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int nx = blockDim.x * gridDim.x;
int ny = blockDim.y * gridDim.y;

int t = threadIdx.x + threadIdx.y * blockDim.x; 
int nt = blockDim.x * blockDim.y; 

int g = blockIdx.x + blockIdx.y * gridDim.x;

unsigned int *gmem = out + g * NUM_PARTS;
for (int i = t; i < ACTIVE_CHANNELS * NUM_BINS; i += nt)
gmem[i] = 0;
__syncthreads();

for (int col = x; col < width; col += nx)
{
for (int row = y; row < height; row += ny)
{
PixelType pixel = in[row * width + col];

unsigned int bins[ACTIVE_CHANNELS];
DecodePixel<NUM_BINS>(pixel, bins);

#pragma unroll
for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
atomicAdd(&gmem[(NUM_BINS * CHANNEL) + bins[CHANNEL]], 1);
}
}
}

template <
int         NUM_PARTS,
int         ACTIVE_CHANNELS,
int         NUM_BINS>
__global__ void histogram_gmem_accum(
const unsigned int *in,
int n,
unsigned int *out)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i > ACTIVE_CHANNELS * NUM_BINS)
return; 

unsigned int total = 0;
for (int j = 0; j < n; j++)
total += in[i + NUM_PARTS * j];

out[i] = total;
}




template <
int         ACTIVE_CHANNELS,
int         NUM_BINS,
typename    PixelType>
double run_gmem_atomics(
PixelType *d_image,
int width,
int height,
unsigned int *d_hist,
bool warmup)
{
enum
{
NUM_PARTS = 1024
};

hipDeviceProp_t props;
hipGetDeviceProperties(&props, 0);

dim3 block(32, 4);
dim3 grid(16, 16);
int total_blocks = grid.x * grid.y;

unsigned int *d_part_hist;
hipMalloc(&d_part_hist, total_blocks * NUM_PARTS * sizeof(unsigned int));

dim3 block2(128);
dim3 grid2((3 * NUM_BINS + block.x - 1) / block.x);

GpuTimer gpu_timer;
gpu_timer.Start();

hipLaunchKernelGGL(HIP_KERNEL_NAME(histogram_gmem_atomics<NUM_PARTS, ACTIVE_CHANNELS, NUM_BINS>), grid, block, 0, 0, 
d_image,
width,
height,
d_part_hist);

hipLaunchKernelGGL(HIP_KERNEL_NAME(histogram_gmem_accum<NUM_PARTS, ACTIVE_CHANNELS, NUM_BINS>), grid2, block2, 0, 0, 
d_part_hist,
total_blocks,
d_hist);

gpu_timer.Stop();
float elapsed_millis = gpu_timer.ElapsedMillis();

hipFree(d_part_hist);

return elapsed_millis;
}

