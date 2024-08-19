
#include <PiTable.hpp>
#include <primecount-internal.hpp>
#include <primesieve.hpp>
#include <pod_vector.hpp>
#include <imath.hpp>
#include <macros.hpp>
#include <min.hpp>

#include <stdint.h>
#include <algorithm>

namespace primecount {

const pod_array<PiTable::pi_t, 64> PiTable::pi_cache_ =
{{
{    3, 0xF93DDBB67EEFDFFEull }, {   52, 0x9EEDA6EAF31E4FD5ull },
{   92, 0xA559DD3BD3D30CE6ull }, {  128, 0x56A61E78BD92676Aull },
{  162, 0x554C2ADE2DADE356ull }, {  196, 0xF8A154039FF0A3D9ull },
{  228, 0x3A13F666E944FD2Eull }, {  263, 0x54BF11453A2B4CB8ull },
{  293, 0x4F8CBCC8B37AC18Cull }, {  325, 0xEF17C19B71715821ull },
{  357, 0x468C83E5081A9654ull }, {  382, 0x87588F9265AEFB72ull },
{  417, 0xA0E3266581D892D2ull }, {  444, 0x99EB813C26C73811ull },
{  473, 0x4D33F3243E88518Dull }, {  503, 0x4C58B42AA71C8B5Aull },
{  532, 0xC383DC8219F6264Eull }, {  562, 0x02CDCDB50238F12Cull },
{  590, 0x307A4C570C944AB2ull }, {  617, 0xF8246C44CBF10B43ull },
{  646, 0x8DEA735CA8950119ull }, {  675, 0xC41E22A6502B9624ull },
{  700, 0x9C742F3AD40648D1ull }, {  729, 0x2E1568BF88056A07ull },
{  757, 0x14089851B7E35560ull }, {  783, 0x2770494D45AA5A86ull },
{  811, 0x618302ABCAD593D2ull }, {  840, 0xADA9C22287CE2405ull },
{  867, 0xB01689D1784D8C18ull }, {  893, 0x522434C0A262C757ull },
{  919, 0x4308218D32405AAEull }, {  942, 0x60E119D9B6D2B634ull },
{  973, 0x947A44D060391A67ull }, { 1000, 0x105574A88388099Aull },
{ 1023, 0x32C8231E685DA127ull }, { 1051, 0x38B14873440319E0ull },
{ 1075, 0x1CB59861572AE6C3ull }, { 1106, 0x2902AC8F81C5680Aull },
{ 1130, 0x2E644E1194E3471Aull }, { 1158, 0x1006C514DC3DCB14ull },
{ 1184, 0xE34730E982B129E9ull }, { 1214, 0xB430300A25C31934ull },
{ 1237, 0x4C8ED84446E5C16Cull }, { 1265, 0x818992787024225Dull },
{ 1289, 0xA508E9861B265682ull }, { 1315, 0x104AC2B029C3D300ull },
{ 1337, 0xC760421DA13859B2ull }, { 1364, 0x8BC61A44C88C2722ull },
{ 1389, 0x0931A610461A8182ull }, { 1409, 0x15A9D8D2182F54F0ull },
{ 1438, 0x91500EC0F60C2E06ull }, { 1462, 0xC319653818C126CDull },
{ 1489, 0x4A84D62D2A8B9356ull }, { 1518, 0xC476E0092CA50A61ull },
{ 1543, 0x1B6614E808D83C6Aull }, { 1570, 0x073110366302A4B0ull },
{ 1592, 0xA08AC312424892D5ull }, { 1615, 0x5C788582A4742D9Full },
{ 1645, 0xE8021D1461B0180Dull }, { 1667, 0x30831C4901C11218ull },
{ 1686, 0xF40C0FD888A13367ull }, { 1715, 0xB1474266D7588898ull },
{ 1743, 0x155941180896A816ull }, { 1765, 0xA1AAB3E1522A44B5ull }
}};

PiTable::PiTable(uint64_t max_x, int threads) :
max_x_(max_x)
{
uint64_t limit = max_x + 1;
pi_.resize(ceil_div(limit, 240));
std::size_t n = min(pi_cache_.size(), pi_.size());
std::copy_n(&pi_cache_[0], n, &pi_[0]);

uint64_t cache_limit = pi_cache_.size() * 240;
if (limit > cache_limit)
init(limit, cache_limit, threads);
}

void PiTable::init(uint64_t limit,
uint64_t cache_limit,
int threads)
{
ASSERT(cache_limit < limit);
uint64_t dist = limit - cache_limit;
uint64_t thread_threshold = (uint64_t) 1e7;
threads = ideal_num_threads(dist, threads, thread_threshold);
uint64_t thread_dist = dist / threads;
thread_dist = max(thread_threshold, thread_dist);
thread_dist += 240 - thread_dist % 240;
counts_.resize(threads);

#pragma omp parallel num_threads(threads)
{
#pragma omp for
for (int t = 0; t < threads; t++)
{
uint64_t low = cache_limit + thread_dist * t;
uint64_t high = low + thread_dist;
high = min(high, limit);

if (low < high)
init_bits(low, high, t);
}

#pragma omp for
for (int t = 0; t < threads; t++)
{
uint64_t low = cache_limit + thread_dist * t;
uint64_t high = low + thread_dist;
high = min(high, limit);

if (low < high)
init_count(low, high, t);
}
}
}

void PiTable::init_bits(uint64_t low,
uint64_t high,
uint64_t thread_num)
{
uint64_t i = low / 240;
uint64_t j = ceil_div(high, 240);
std::fill_n(&pi_[i], j - i, pi_t{0, 0});

low = max(low, 7);
primesieve::iterator it(low, high);
uint64_t count = 0;
uint64_t prime = 0;

while ((prime = it.next_prime()) < high)
{
uint64_t prime_bit = set_bit_[prime % 240];
pi_[prime / 240].bits |= prime_bit;
count += 1;
}

counts_[thread_num] = count;
}

void PiTable::init_count(uint64_t low,
uint64_t high,
uint64_t thread_num)
{
pi_t cache_last = pi_cache_.back();
uint64_t count = cache_last.count + popcnt64(cache_last.bits);
for (uint64_t i = 0; i < thread_num; i++)
count += counts_[i];

uint64_t i = low / 240;
uint64_t stop_idx = ceil_div(high, 240);

for (; i < stop_idx; i++)
{
pi_[i].count = count;
count += popcnt64(pi_[i].bits);
}
}

} 