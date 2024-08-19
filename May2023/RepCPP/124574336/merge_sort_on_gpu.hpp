
#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_MERGE_SORT_ON_GPU_HPP_
#define BOOST_COMPUTE_ALGORITHM_DETAIL_MERGE_SORT_ON_GPU_HPP_

#include <algorithm>

#include <boost/compute/kernel.hpp>
#include <boost/compute/program.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/memory/local_buffer.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class KeyType, class ValueType>
inline size_t pick_bitonic_block_sort_block_size(size_t proposed_wg,
size_t lmem_size,
bool sort_by_key)
{
size_t n = proposed_wg;

size_t lmem_required = n * sizeof(KeyType);
if(sort_by_key) {
lmem_required += n * sizeof(ValueType);
}

while(lmem_size < (lmem_required * 4) && (n > 64)) {
n /= 2;
lmem_required = n * sizeof(KeyType);
}
while(lmem_size < lmem_required && (n != 1)) {
n /= 2;
if(n < 1) n = 1;
lmem_required = n * sizeof(KeyType);
}

if(n < 2)   { return 1; }
else if(n < 4)   { return 2; }
else if(n < 8)   { return 4; }
else if(n < 16)  { return 8; }
else if(n < 32)  { return 16; }
else if(n < 64)  { return 32; }
else if(n < 128) { return 64; }
else if(n < 256) { return 128; }
else             { return 256; }
}


template<class KeyIterator, class ValueIterator, class Compare>
inline size_t bitonic_block_sort(KeyIterator keys_first,
ValueIterator values_first,
Compare compare,
const size_t count,
const bool sort_by_key,
command_queue &queue)
{
typedef typename std::iterator_traits<KeyIterator>::value_type key_type;
typedef typename std::iterator_traits<ValueIterator>::value_type value_type;

meta_kernel k("bitonic_block_sort");
size_t count_arg = k.add_arg<const uint_>("count");

size_t local_keys_arg = k.add_arg<key_type *>(memory_object::local_memory, "lkeys");
size_t local_vals_arg = 0;
if(sort_by_key) {
local_vals_arg = k.add_arg<uchar_ *>(memory_object::local_memory, "lidx");
}

k <<
k.decl<const uint_>("gid") << " = get_global_id(0);\n" <<
k.decl<const uint_>("lid") << " = get_local_id(0);\n";

k <<
k.decl<key_type>("my_key") << ";\n";
if(sort_by_key)
{
k <<
k.decl<uchar_>("my_index") << " = (uchar)(lid);\n";
}

k <<
"if(gid < count) {\n" <<
k.var<key_type>("my_key") <<  " = " <<
keys_first[k.var<const uint_>("gid")] << ";\n" <<
"}\n";

k <<
"lkeys[lid] = my_key;\n";
if(sort_by_key)
{
k <<
"lidx[lid] = my_index;\n";
}
k <<
k.decl<const uint_>("offset") << " = get_group_id(0) * get_local_size(0);\n" <<
k.decl<const uint_>("n") << " = min((uint)(get_local_size(0)),(count - offset));\n";


k <<
"if(((n != 0) && ((n & (~n + 1)) == n))) {\n";

k <<
"barrier(CLK_LOCAL_MEM_FENCE);\n" <<

"#pragma unroll\n" <<
"for(" <<
k.decl<uint_>("length") << " = 1; " <<
"length < n; " <<
"length <<= 1" <<
") {\n" <<
k.decl<bool>("direction") << "= ((lid & (length<<1)) != 0);\n" <<
"for(" <<
k.decl<uint_>("k") << " = length; " <<
"k > 0; " <<
"k >>= 1" <<
") {\n" <<

k.decl<uint_>("sibling_idx") << " = lid ^ k;\n" <<
k.decl<key_type>("sibling_key") << " = lkeys[sibling_idx];\n" <<
k.decl<bool>("compare") << " = " <<
compare(k.var<key_type>("sibling_key"),
k.var<key_type>("my_key")) << ";\n" <<
k.decl<bool>("equal") << " = !(compare || " <<
compare(k.var<key_type>("my_key"),
k.var<key_type>("sibling_key")) << ");\n" <<
k.decl<bool>("swap") <<
" = compare ^ (sibling_idx < lid) ^ direction;\n" <<
"swap = equal ? false : swap;\n" <<
"my_key = swap ? sibling_key : my_key;\n";
if(sort_by_key)
{
k <<
"my_index = swap ? lidx[sibling_idx] : my_index;\n";
}
k <<
"barrier(CLK_LOCAL_MEM_FENCE);\n" <<
"lkeys[lid] = my_key;\n";
if(sort_by_key)
{
k <<
"lidx[lid] = my_index;\n";
}
k <<
"barrier(CLK_LOCAL_MEM_FENCE);\n" <<
"}\n" <<
"}\n";


k <<
"}\n" <<
"else { \n";

k <<
k.decl<bool>("lid_is_even") << " = (lid%2) == 0;\n" <<
k.decl<uint_>("oddsibling_idx") << " = " <<
"(lid_is_even) ? max(lid,(uint)(1)) - 1 : min(lid+1,n-1);\n" <<
k.decl<uint_>("evensibling_idx") << " = " <<
"(lid_is_even) ? min(lid+1,n-1) : max(lid,(uint)(1)) - 1;\n" <<

"barrier(CLK_LOCAL_MEM_FENCE);\n" <<

"#pragma unroll\n" <<
"for(" <<
k.decl<uint_>("i") << " = 0; " <<
"i < n; " <<
"i++" <<
") {\n" <<
k.decl<uint_>("sibling_idx") <<
" = i%2 == 0 ? evensibling_idx : oddsibling_idx;\n" <<
k.decl<key_type>("sibling_key") << " = lkeys[sibling_idx];\n" <<
k.decl<bool>("compare") << " = " <<
compare(k.var<key_type>("sibling_key"),
k.var<key_type>("my_key")) << ";\n" <<
k.decl<bool>("equal") << " = !(compare || " <<
compare(k.var<key_type>("my_key"),
k.var<key_type>("sibling_key")) << ");\n" <<
k.decl<bool>("swap") <<
" = compare ^ (sibling_idx < lid);\n" <<
"swap = equal ? false : swap;\n" <<
"my_key = swap ? sibling_key : my_key;\n";
if(sort_by_key)
{
k <<
"my_index = swap ? lidx[sibling_idx] : my_index;\n";
}
k <<
"barrier(CLK_LOCAL_MEM_FENCE);\n" <<
"lkeys[lid] = my_key;\n";
if(sort_by_key)
{
k <<
"lidx[lid] = my_index;\n";
}
k <<
"barrier(CLK_LOCAL_MEM_FENCE);\n"
"}\n" <<  

"}\n"; 

k <<
"if(gid < count) {\n" <<
keys_first[k.var<const uint_>("gid")] << " = " <<
k.var<key_type>("my_key") << ";\n";
if(sort_by_key)
{
k <<
k.decl<value_type>("my_value") << " = " <<
values_first[k.var<const uint_>("offset + my_index")] << ";\n" <<
"barrier(CLK_GLOBAL_MEM_FENCE);\n" <<
values_first[k.var<const uint_>("gid")] << " = my_value;\n";
}
k <<
"}\n";

const context &context = queue.get_context();
const device &device = queue.get_device();
::boost::compute::kernel kernel = k.compile(context);

const size_t work_group_size =
pick_bitonic_block_sort_block_size<key_type, uchar_>(
kernel.get_work_group_info<size_t>(
device, CL_KERNEL_WORK_GROUP_SIZE
),
device.get_info<size_t>(CL_DEVICE_LOCAL_MEM_SIZE),
sort_by_key
);

const size_t global_size =
work_group_size * static_cast<size_t>(
std::ceil(float(count) / work_group_size)
);

kernel.set_arg(count_arg, static_cast<uint_>(count));
kernel.set_arg(local_keys_arg, local_buffer<key_type>(work_group_size));
if(sort_by_key) {
kernel.set_arg(local_vals_arg, local_buffer<uchar_>(work_group_size));
}

queue.enqueue_1d_range_kernel(kernel, 0, global_size, work_group_size);
return work_group_size;
}

template<class KeyIterator, class ValueIterator, class Compare>
inline size_t block_sort(KeyIterator keys_first,
ValueIterator values_first,
Compare compare,
const size_t count,
const bool sort_by_key,
const bool stable,
command_queue &queue)
{
if(stable) {
return size_t(1);
}
return bitonic_block_sort(
keys_first, values_first,
compare, count,
sort_by_key, queue
);
}

template<class KeyIterator, class ValueIterator, class Compare>
inline void merge_blocks_on_gpu(KeyIterator keys_first,
ValueIterator values_first,
KeyIterator out_keys_first,
ValueIterator out_values_first,
Compare compare,
const size_t count,
const size_t block_size,
const bool sort_by_key,
command_queue &queue)
{
typedef typename std::iterator_traits<KeyIterator>::value_type key_type;
typedef typename std::iterator_traits<ValueIterator>::value_type value_type;

meta_kernel k("merge_blocks");
size_t count_arg = k.add_arg<const uint_>("count");
size_t block_size_arg = k.add_arg<const uint_>("block_size");

k <<
k.decl<const uint_>("gid") << " = get_global_id(0);\n" <<
"if(gid >= count) {\n" <<
"return;\n" <<
"}\n" <<

k.decl<const key_type>("my_key") << " = " <<
keys_first[k.var<const uint_>("gid")] << ";\n";

if(sort_by_key) {
k <<
k.decl<const value_type>("my_value") << " = " <<
values_first[k.var<const uint_>("gid")] << ";\n";
}

k <<
k.decl<const uint_>("my_block_idx") << " = gid / block_size;\n" <<
k.decl<const bool>("my_block_idx_is_odd") << " = " <<
"my_block_idx & 0x1;\n" <<

k.decl<const uint_>("other_block_idx") << " = " <<
"my_block_idx_is_odd ? my_block_idx - 1 : my_block_idx + 1;\n" <<

k.decl<const uint_>("my_block_start") << " = " <<
"min(my_block_idx * block_size, count);\n" << 
k.decl<const uint_>("my_block_end") << " = " <<
"min((my_block_idx + 1) * block_size, count);\n" << 

k.decl<const uint_>("other_block_start") << " = " <<
"min(other_block_idx * block_size, count);\n" << 
k.decl<const uint_>("other_block_end") << " = " <<
"min((other_block_idx + 1) * block_size, count);\n" << 

"if(other_block_start == count){\n" <<
out_keys_first[k.var<uint_>("gid")] << " = my_key;\n";
if(sort_by_key) {
k <<
out_values_first[k.var<uint_>("gid")] << " = my_value;\n";
}

k <<
"return;\n" <<
"}\n" <<

k.decl<uint_>("left_idx") << " = other_block_start;\n" <<
k.decl<uint_>("right_idx") << " = other_block_end;\n" <<
"while(left_idx < right_idx) {\n" <<
k.decl<uint_>("mid_idx") << " = (left_idx + right_idx) / 2;\n" <<
k.decl<key_type>("mid_key") << " = " <<
keys_first[k.var<const uint_>("mid_idx")] << ";\n" <<
k.decl<bool>("smaller") << " = " <<
compare(k.var<key_type>("mid_key"),
k.var<key_type>("my_key")) << ";\n" <<
"left_idx = smaller ? mid_idx + 1 : left_idx;\n" <<
"right_idx = smaller ? right_idx :  mid_idx;\n" <<
"}\n" <<

"right_idx = other_block_end;\n" <<
"if(my_block_idx_is_odd && left_idx != right_idx) {\n" <<
k.decl<key_type>("upper_key") << " = " <<
keys_first[k.var<const uint_>("left_idx")] << ";\n" <<
"while(" <<
"!(" << compare(k.var<key_type>("upper_key"),
k.var<key_type>("my_key")) <<
") && " <<
"!(" << compare(k.var<key_type>("my_key"),
k.var<key_type>("upper_key")) <<
") && " <<
"left_idx < right_idx" <<
")" <<
"{\n" <<
k.decl<uint_>("mid_idx") << " = (left_idx + right_idx) / 2;\n" <<
k.decl<key_type>("mid_key") << " = " <<
keys_first[k.var<const uint_>("mid_idx")] << ";\n" <<
k.decl<bool>("equal") << " = " <<
"!(" << compare(k.var<key_type>("mid_key"),
k.var<key_type>("my_key")) <<
") && " <<
"!(" << compare(k.var<key_type>("my_key"),
k.var<key_type>("mid_key")) <<
");\n" <<
"left_idx = equal ? mid_idx + 1 : left_idx + 1;\n" <<
"right_idx = equal ? right_idx : mid_idx;\n" <<
"upper_key = " <<
keys_first[k.var<const uint_>("left_idx")] << ";\n" <<
"}\n" <<
"}\n" <<

k.decl<uint_>("offset") << " = 0;\n" <<
"offset += gid - my_block_start;\n" <<
"offset += left_idx - other_block_start;\n" <<
"offset += min(my_block_start, other_block_start);\n" <<
out_keys_first[k.var<uint_>("offset")] << " = my_key;\n";
if(sort_by_key) {
k <<
out_values_first[k.var<uint_>("offset")] << " = my_value;\n";
}

const context &context = queue.get_context();
::boost::compute::kernel kernel = k.compile(context);

const size_t work_group_size = (std::min)(
size_t(256),
kernel.get_work_group_info<size_t>(
queue.get_device(), CL_KERNEL_WORK_GROUP_SIZE
)
);
const size_t global_size =
work_group_size * static_cast<size_t>(
std::ceil(float(count) / work_group_size)
);

kernel.set_arg(count_arg, static_cast<uint_>(count));
kernel.set_arg(block_size_arg, static_cast<uint_>(block_size));
queue.enqueue_1d_range_kernel(kernel, 0, global_size, work_group_size);
}

template<class KeyIterator, class ValueIterator, class Compare>
inline void merge_sort_by_key_on_gpu(KeyIterator keys_first,
KeyIterator keys_last,
ValueIterator values_first,
Compare compare,
bool stable,
command_queue &queue)
{
typedef typename std::iterator_traits<KeyIterator>::value_type key_type;
typedef typename std::iterator_traits<ValueIterator>::value_type value_type;

size_t count = iterator_range_size(keys_first, keys_last);
if(count < 2){
return;
}

size_t block_size =
block_sort(
keys_first, values_first,
compare, count,
true , stable ,
queue
);

if(count <= block_size) {
return;
}

const context &context = queue.get_context();

bool result_in_temporary_buffer = false;
::boost::compute::vector<key_type> temp_keys(count, context);
::boost::compute::vector<value_type> temp_values(count, context);

for(; block_size < count; block_size *= 2) {
result_in_temporary_buffer = !result_in_temporary_buffer;
if(result_in_temporary_buffer) {
merge_blocks_on_gpu(keys_first, values_first,
temp_keys.begin(), temp_values.begin(),
compare, count, block_size,
true , queue);
} else {
merge_blocks_on_gpu(temp_keys.begin(), temp_values.begin(),
keys_first, values_first,
compare, count, block_size,
true , queue);
}
}

if(result_in_temporary_buffer) {
copy_async(temp_keys.begin(), temp_keys.end(), keys_first, queue);
copy_async(temp_values.begin(), temp_values.end(), values_first, queue);
}
}

template<class Iterator, class Compare>
inline void merge_sort_on_gpu(Iterator first,
Iterator last,
Compare compare,
bool stable,
command_queue &queue)
{
typedef typename std::iterator_traits<Iterator>::value_type key_type;

size_t count = iterator_range_size(first, last);
if(count < 2){
return;
}

Iterator dummy;
size_t block_size =
block_sort(
first, dummy,
compare, count,
false , stable ,
queue
);

if(count <= block_size) {
return;
}

const context &context = queue.get_context();

bool result_in_temporary_buffer = false;
::boost::compute::vector<key_type> temp_keys(count, context);

for(; block_size < count; block_size *= 2) {
result_in_temporary_buffer = !result_in_temporary_buffer;
if(result_in_temporary_buffer) {
merge_blocks_on_gpu(first, dummy, temp_keys.begin(), dummy,
compare, count, block_size,
false , queue);
} else {
merge_blocks_on_gpu(temp_keys.begin(), dummy, first, dummy,
compare, count, block_size,
false , queue);
}
}

if(result_in_temporary_buffer) {
copy_async(temp_keys.begin(), temp_keys.end(), first, queue);
}
}

template<class KeyIterator, class ValueIterator, class Compare>
inline void merge_sort_by_key_on_gpu(KeyIterator keys_first,
KeyIterator keys_last,
ValueIterator values_first,
Compare compare,
command_queue &queue)
{
merge_sort_by_key_on_gpu(
keys_first, keys_last, values_first,
compare, false , queue
);
}

template<class Iterator, class Compare>
inline void merge_sort_on_gpu(Iterator first,
Iterator last,
Compare compare,
command_queue &queue)
{
merge_sort_on_gpu(
first, last, compare, false , queue
);
}

} 
} 
} 

#endif 
