
template <
int         ACTIVE_CHANNELS,
int         NUM_BINS,
typename    PixelType>
double run_gmem_atomics(
queue &q, 
buffer<PixelType,1> &d_image,
int width,
int height,
buffer<unsigned int,1> &d_hist,
bool warmup)
{
enum
{
NUM_PARTS = 1024
};

const int total_blocks = 256;

buffer<unsigned int, 1> d_part_hist (total_blocks * NUM_PARTS);

auto start = std::chrono::steady_clock::now();

q.submit([&] (handler& cgh) {
auto in = d_image.template get_access<sycl_read>(cgh);
auto out = d_part_hist.get_access<sycl_atomic>(cgh);
cgh.parallel_for<class hist_gmem_atomics<ACTIVE_CHANNELS, NUM_BINS, PixelType>>(
nd_range<2>(range<2>(64, 512), range<2>(4, 32)), [=] (nd_item<2> item) {
int x = item.get_global_id(1);
int y = item.get_global_id(0);
int nx = item.get_global_range(1);
int ny = item.get_global_range(0);
int t = item.get_local_linear_id();
int nt = item.get_local_range(1)*item.get_local_range(0);
int g = item.get_group_linear_id(); 

for (int i = t; i < ACTIVE_CHANNELS * NUM_BINS; i += nt)
out[g * NUM_PARTS + i].store(0);

item.barrier(access::fence_space::global_and_local);

for (int col = x; col < width; col += nx)
{
for (int row = y; row < height; row += ny)
{
PixelType pixel = in[row * width + col];

unsigned int bins[ACTIVE_CHANNELS];
DecodePixel<NUM_BINS>(pixel, bins);

#pragma unroll
for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
atomic_fetch_add(out[g * NUM_PARTS + (NUM_BINS * CHANNEL) + bins[CHANNEL]], 1U);
}
}

});
});

#ifdef DEBUG
printf("partial histogram:\n");
auto h_part_hist =  d_part_hist.get_access<sycl_read>();
for (int i = 0; i < total_blocks * NUM_PARTS; i++) {
printf("%u\n", h_part_hist[i]);
}
#endif

q.submit([&] (handler& cgh) {
auto in = d_part_hist.get_access<sycl_read>(cgh);
auto out = d_hist.get_access<sycl_write>(cgh);
cgh.parallel_for<class hist_gmem_accum<ACTIVE_CHANNELS, NUM_BINS, PixelType>>(
nd_range<1>(range<1>((ACTIVE_CHANNELS * NUM_BINS + 127) / 128 * 128), range<1>(128)), [=] (
nd_item<1> item) {
int i = item.get_global_id(0);
if (i > ACTIVE_CHANNELS * NUM_BINS) return; 

unsigned int total = 0;
for (int j = 0; j < total_blocks; j++)
total += in[i + NUM_PARTS * j];

out[i] = total;
});
});

q.wait();

auto end = std::chrono::steady_clock::now();
std::chrono::duration<double> elapsed_seconds = end-start;
float elapsed_millis = elapsed_seconds.count()  * 1000;

return elapsed_millis;
}

