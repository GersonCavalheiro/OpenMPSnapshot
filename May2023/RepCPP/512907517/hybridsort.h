#ifndef HYBRIDSORT_H
#define HYBRIDSORT_H

#include <algorithm>    
#include <cmath>
#include <cstddef>    
#include <functional>
#include <omp.h>
#include <type_traits>
#include <utility>       

namespace parallel_stable_sorting {

constexpr int INSERTION_SORT_MAXLEN = 32; 
constexpr int K_THREADS_TASK = 64; 

constexpr int IN_PLACE_INSERTION_SORT_MAXLEN = 32; 
constexpr int INPLACE_K_THREADS_TASK = 8; 

constexpr int EXCHANGE_BLOCK_SWAP_BLOCK_LENGTH_MIN = 512; 

namespace util_funcs {

inline int upper_power_of_two(int i) {
--i; 
i |= i >> 1; 
i |= i >> 2;
i |= i >> 4;
i |= i >> 8;
i |= i >> 16;
return i + 1;
}

inline int lower_power_of_two(int i) {
i |= i >> 1; 
i |= i >> 2;
i |= i >> 4;
i |= i >> 8;
i |= i >> 16;
return (i + 1) >> 1;  
}

template <class RandomAccessIteratorA, class RandomAccessIteratorB, class Comparator>
inline void SplitEvenly(Comparator compare, RandomAccessIteratorA left_begin, std::size_t left_length, RandomAccessIteratorB right_begin, std::size_t right_length, RandomAccessIteratorA& left_middle, RandomAccessIteratorB& right_middle) {
if (left_length < right_length) {
right_begin += (right_length - left_length) / 2;
}
else {
left_begin += (left_length - right_length) / 2;
left_length = right_length;
}
while (left_length) {
if (compare(right_begin[left_length / 2], left_begin[(left_length - 1) / 2])) {
right_begin += (left_length + 2) / 2;
left_length = (left_length - 1) / 2;
}
else {
left_begin += (left_length + 1) / 2;
left_length /= 2;
}
}
left_middle = left_begin;
right_middle = right_begin;
}

}  
using namespace util_funcs;

namespace block_operations {
template <class RandomAccessIteratorA, class RandomAccessIteratorB>
inline void MoveBlock(RandomAccessIteratorA src, RandomAccessIteratorB dest, std::size_t length) {
for (std::size_t i = 0; i < length; ++i) dest[i] = std::move(src[i]);
}

template <class RandomAccessIteratorA, class RandomAccessIteratorB>
inline void SwapBlocks(RandomAccessIteratorA left_begin, RandomAccessIteratorB right_begin, std::size_t length) {
for (std::size_t i = 0; i < length; ++i) std::swap(left_begin[i], right_begin[i]);
}

template <class RandomAccessIteratorA, class RandomAccessIteratorB>
inline void ReverseSwapBlocks(RandomAccessIteratorA left_begin, RandomAccessIteratorB right_end, std::size_t length) {
for (std::size_t i = 0; i < length; ++i) std::swap(left_begin[i], *--right_end);
}

template <class RandomAccessIteratorA, class RandomAccessIteratorB>
inline void Reverse(RandomAccessIteratorA begin, RandomAccessIteratorB end) {
while (begin < --end) std::swap(*begin++, *end);
}

template <class RandomAccessIteratorA, class RandomAccessIteratorB>
inline void ReversingRotateBlocks(RandomAccessIteratorA left_begin, RandomAccessIteratorA left_end, RandomAccessIteratorB right_begin, RandomAccessIteratorB right_end) {
if (left_begin != left_end && right_begin != right_end) {
Reverse(left_begin, left_end);
Reverse(right_begin, right_end);
Reverse(left_begin, right_end);
}
}

template <class RandomAccessIteratorA, class RandomAccessIteratorB>
inline void ExchangeRotateBlocks(RandomAccessIteratorA left_begin, std::size_t left_length, RandomAccessIteratorB right_begin, std::size_t right_length) {
while (left_length > EXCHANGE_BLOCK_SWAP_BLOCK_LENGTH_MIN && right_length > EXCHANGE_BLOCK_SWAP_BLOCK_LENGTH_MIN) {
if (left_length < right_length) {
SwapBlocks(left_begin, right_begin + right_length - left_length, left_length);
right_length -= left_length;
}
else {
SwapBlocks(left_begin, right_begin, right_length);
left_length -= right_length;
left_begin += right_length;
}
}
ReversingRotateBlocks(left_begin, left_begin + left_length, right_begin, right_begin + right_length);
}

template <class RandomAccessIteratorA, class RandomAccessIteratorB>
inline void HalfReverseSwapBlocks(RandomAccessIteratorA keep_begin, RandomAccessIteratorB reverse_begin, std::size_t length) {
typename std::remove_reference<decltype(*reverse_begin)>::type tmp;
RandomAccessIteratorA keep_last = keep_begin + length - 1, reverse_last = reverse_begin + length - 1;
while (keep_begin < keep_last) {
tmp = std::move(*reverse_begin);
*reverse_begin = std::move(*keep_begin);
*keep_begin = std::move(*reverse_last);
*reverse_last = std::move(*keep_last);
*keep_last = std::move(tmp);
++keep_begin, ++reverse_begin;
--keep_last, --reverse_last;
}
if (keep_begin == keep_last) std::swap(*keep_begin, *reverse_begin);
}

template <class RandomAccessIteratorA, class RandomAccessIteratorB>
inline void ReversingExchangeRotateBlocks(RandomAccessIteratorA left_begin, std::size_t left_length, RandomAccessIteratorB right_begin, std::size_t right_length) {
if (left_length > EXCHANGE_BLOCK_SWAP_BLOCK_LENGTH_MIN && right_length > EXCHANGE_BLOCK_SWAP_BLOCK_LENGTH_MIN) for (;;) {
if (left_length < right_length) {
right_length -= left_length;
if (right_length > EXCHANGE_BLOCK_SWAP_BLOCK_LENGTH_MIN) {
SwapBlocks(left_begin, right_begin + right_length, left_length);
}
else {
HalfReverseSwapBlocks(left_begin, right_begin + right_length, left_length);
Reverse(right_begin, right_begin + right_length);
Reverse(left_begin, right_begin + right_length);
return;
}
}
else {
left_length -= right_length;
if (left_length > EXCHANGE_BLOCK_SWAP_BLOCK_LENGTH_MIN) {
SwapBlocks(left_begin, right_begin, right_length);
}
else if (left_length == 0) {
SwapBlocks(left_begin, right_begin, right_length);
return;
}
else {
HalfReverseSwapBlocks(right_begin, left_begin, right_length);
left_begin += right_length;
Reverse(left_begin, left_begin + left_length);
Reverse(left_begin, right_begin + right_length);
return;
}
left_begin += right_length;
}
}
else {
ReversingRotateBlocks(left_begin, left_begin + left_length, right_begin, right_begin + right_length);
}
}

template <class RandomAccessIteratorA, class RandomAccessIteratorB>
inline void RotateBlocks(RandomAccessIteratorA left_begin, RandomAccessIteratorA left_end, RandomAccessIteratorB right_begin, RandomAccessIteratorB right_end) {
ReversingExchangeRotateBlocks(left_begin, left_end - left_begin, right_begin, right_end - left_end);
}

template <class RandomAccessIteratorA, class RandomAccessIteratorB>
inline void ParallelReverse(RandomAccessIteratorA begin, RandomAccessIteratorB end, int tasks) {
std::size_t length = (end - begin) / 2;
if (length < tasks) std::reverse(begin, end);
else {
std::size_t tasklength = length / tasks;
for (int i = 1; i < tasks; ++i) {
#pragma omp task
ReverseSwapBlocks(begin, end, tasklength);
begin += tasklength;
end -= tasklength;
}
std::reverse(begin, end);
#pragma omp taskwait
}
}

template <class RandomAccessIteratorA, class RandomAccessIteratorB>
inline void ParallelReversingRotateBlocks(RandomAccessIteratorA left_begin, RandomAccessIteratorA left_end, RandomAccessIteratorB right_begin, RandomAccessIteratorB right_end, int tasks) {
#pragma omp task
ParallelReverse(left_begin, left_end, tasks / 2);
ParallelReverse(right_begin, right_end, tasks / 2);
#pragma omp taskwait
ParallelReverse(left_begin, right_end, tasks);
}

template <class RandomAccessIteratorA, class RandomAccessIteratorB>
inline void ParallelRotateBlocks(RandomAccessIteratorA left_begin, RandomAccessIteratorA left_end, RandomAccessIteratorB right_begin, RandomAccessIteratorB right_end, int tasks) {
ParallelReversingRotateBlocks(left_begin, left_end, right_begin, right_end, tasks);
}

}  
using namespace block_operations;

namespace insertion_sort {

template <class RandomAccessIterator, class Comparator>
inline void InsertionSort(Comparator compare, RandomAccessIterator begin, RandomAccessIterator end) {
typename std::remove_reference<decltype(*begin)>::type tmp;
RandomAccessIterator to_insert = begin + 1;
while (to_insert < end) {
RandomAccessIterator to_compare = to_insert - 1;
if (compare(*to_insert, *to_compare)) {
tmp = std::move(*to_insert);
*to_insert = std::move(*to_compare);
while (--to_compare >= begin && compare(tmp, *to_compare)) {
to_compare[1] = std::move(*to_compare);
}
to_compare[1] = std::move(tmp);
}
++to_insert;
}
}

template <class RandomAccessIterator, class Comparator>
inline void InsertionSort(Comparator compare, RandomAccessIterator begin, RandomAccessIterator end, RandomAccessIterator to_insert) {
typename std::remove_reference<decltype(*begin)>::type tmp;
while (to_insert < end) {
RandomAccessIterator to_compare = to_insert - 1;
if (compare(*to_insert, *to_compare)) {
tmp = std::move(*to_insert);
*to_insert = std::move(*to_compare);
while (--to_compare >= begin && compare(tmp, *to_compare)) {
to_compare[1] = std::move(*to_compare);
}
to_compare[1] = std::move(tmp);
}
++to_insert;
}
}

}  
using namespace insertion_sort;

namespace buffer_extraction {

template <class RandomAccessIterator, class Comparator>
inline std::size_t ExtractBufferBack(Comparator compare, RandomAccessIterator begin, RandomAccessIterator end, std::size_t buffer_length) {
if (end - begin == 0) return 0;
std::swap(begin, end);
--begin; --end;
std::size_t max_length = 1;
RandomAccessIterator next = begin - 1, buffer_begin = begin;
while (max_length < buffer_length) {
if (end < next && !(compare(*next, next[1]))) {
do --next; while (end < next && !(compare(*next, next[1])));
if (next == end) break;
RotateBlocks(next + 1, buffer_begin, buffer_begin, buffer_begin + max_length);
}
if (next == end) break;
++max_length;
buffer_begin = next;
--next;
}
RotateBlocks(buffer_begin, buffer_begin + max_length, buffer_begin + max_length, begin + 1);
return max_length;
}

template <class RandomAccessIterator, class Comparator>
inline std::size_t ExtractBuffer(Comparator compare, RandomAccessIterator begin, RandomAccessIterator end, std::size_t buffer_length) {
if (end - begin == 0) return 0;
std::size_t max_length = 1;
RandomAccessIterator next = begin + 1, buffer_begin = begin;
while (max_length < buffer_length) {
if (next < end && !(compare(next[-1], *next))) {
do ++next; while (next < end && !(compare(next[-1], *next)));
if (next == end) break;
RotateBlocks(buffer_begin, buffer_begin + max_length, buffer_begin + max_length, next);
buffer_begin = next - max_length;
}
if (next == end) break;
++max_length;
++next;
}
RotateBlocks(begin, buffer_begin, buffer_begin, buffer_begin + max_length);
return max_length;
}

template <class RandomAccessIterator, class Comparator>
inline std::size_t ExtractBufferAssorted(Comparator compare, RandomAccessIterator begin, RandomAccessIterator end, std::size_t buffer_length) {
if (end - begin == 0) return 0;
std::size_t max_length = 1;
RandomAccessIterator next = begin + 1, buffer_begin = begin;
while (max_length < buffer_length) {
while (next < end && std::binary_search(buffer_begin, buffer_begin + max_length, *next, compare)) ++next;
if (next == end) break;
RotateBlocks(buffer_begin, buffer_begin + max_length, buffer_begin + max_length, next);
buffer_begin = next - max_length;
++max_length;
InsertionSort(compare, buffer_begin, buffer_begin + max_length, next);
++next;
}
RotateBlocks(begin, buffer_begin, buffer_begin, buffer_begin + max_length);
return max_length;
}

}  
using namespace buffer_extraction;

namespace inplace_merge {

template <class RandomAccessIterator, class SortedTypePointer, class Comparator>
inline void InternalBufferMerge(Comparator compare, SortedTypePointer left_begin, SortedTypePointer left_end, RandomAccessIterator right_begin, RandomAccessIterator right_end, RandomAccessIterator buffer) {
if (left_begin < left_end) {
typename std::remove_reference<decltype(*buffer)>::type tmp = std::move(*buffer);
if (right_begin < right_end) for (;;) {
if (compare(*right_begin, *left_begin)) {
*buffer = std::move(*right_begin);
++buffer;
*right_begin = std::move(*buffer);
++right_begin;
if (right_begin == right_end) break;
}
else {
*buffer = std::move(*left_begin);
++buffer;
*left_begin = std::move(*buffer);
++left_begin;
if (left_begin == left_end) {
*buffer = std::move(left_begin[-1]);
break;
}
}
}
if (left_begin < left_end) for (;;) {
*buffer = std::move(*left_begin);
++left_begin;
++buffer;
if (left_begin == left_end) break;
left_begin[-1] = std::move(*buffer);
}
left_begin[-1] = std::move(tmp);
}
}

template <class RandomAccessIterator, class Comparator>
inline void RotationMerge(Comparator compare, RandomAccessIterator left_begin, std::size_t left_length, RandomAccessIterator right_begin, std::size_t right_length) {
while (left_length && right_length) {
RandomAccessIterator breakpoint = std::lower_bound(right_begin, right_begin + right_length, *left_begin, compare);
std::size_t length = breakpoint - right_begin;
if (length) {
RotateBlocks(left_begin, left_begin + left_length, right_begin, breakpoint);
left_begin += length;
right_begin = breakpoint;
right_length -= length;
}
++left_begin;
--left_length;
}
}

template <class RandomAccessIterator, class Comparator>
inline void SmallBufferRotationMerge(Comparator compare, RandomAccessIterator left_begin, std::size_t left_length, RandomAccessIterator right_begin, std::size_t right_length, RandomAccessIterator buffer_begin, std::size_t buffer_length) {
while (buffer_length < left_length) {
RandomAccessIterator breakpoint = std::lower_bound(right_begin, right_begin + right_length, left_begin[buffer_length - 1], compare);
std::size_t length = breakpoint - right_begin;
if (length) {
RotateBlocks(left_begin + buffer_length, left_begin + left_length, right_begin, breakpoint);
SwapBlocks(left_begin, buffer_begin, buffer_length);
InternalBufferMerge(compare, buffer_begin, buffer_begin + buffer_length, left_begin + buffer_length, left_begin + buffer_length + length, left_begin);
left_begin += length;
right_begin = breakpoint;
right_length -= length;
}
left_begin += buffer_length;
left_length -= buffer_length;
}
if (right_length) {
SwapBlocks(left_begin, buffer_begin, left_length);
InternalBufferMerge(compare, buffer_begin, buffer_begin + left_length, right_begin, right_begin + right_length, left_begin);
}
}

template <class RandomAccessIterator, class Comparator>
inline std::size_t FindMin(Comparator compare, RandomAccessIterator begin, std::size_t length) {
std::size_t min = 0;
for (std::size_t i = 1; i < length; ++i) if (compare(begin[i], begin[min])) min = i;
return min;
}

template <class RandomAccessIterator, class Comparator>
inline std::size_t RearrangeBlocks(Comparator compare, RandomAccessIterator s1_begin, RandomAccessIterator s2_begin, RandomAccessIterator w_begin, std::size_t w_num, RandomAccessIterator v_begin, std::size_t v_num, RandomAccessIterator mi_begin, std::size_t block_length) {
typename std::remove_reference<decltype(*mi_begin)>::type tmp;
std::size_t w = 0, bds_id = 0;
while (w_num) {
if (v_num && compare(v_begin[block_length - 1], w_begin[w * block_length])) {
SwapBlocks(w_begin, v_begin, block_length);
tmp = std::move(mi_begin[0]);
for (std::size_t i = 1; i < w_num; ++i) mi_begin[i - 1] = std::move(mi_begin[i]);
mi_begin[w_num - 1] = std::move(tmp);
if (w == 0) w = w_num - 1;
else --w;
--v_num;
v_begin += block_length;
}
else {
if (w != 0) {
SwapBlocks(w_begin, w_begin + w * block_length, block_length);
std::swap(mi_begin[0], mi_begin[w]);
}
--w_num;
++mi_begin;
w = FindMin(compare, mi_begin, w_num);
std::swap(s1_begin[bds_id], s2_begin[bds_id]);
}
w_begin += block_length;
++bds_id;
}
return bds_id;
}

template <class RandomAccessIterator, class Comparator>
inline RandomAccessIterator MergeBlocks(Comparator compare, RandomAccessIterator s1_begin, RandomAccessIterator s2_begin, RandomAccessIterator left_begin, RandomAccessIterator right_end, RandomAccessIterator buffer_begin, std::size_t buffer_length, std::size_t bds_id, std::size_t block_length, bool enough_buffer) {
RandomAccessIterator right_begin = left_begin + bds_id * block_length;
while (bds_id--) {
if (compare(s1_begin[bds_id], s2_begin[bds_id])) {
right_begin = left_begin + bds_id * block_length;
}
else {
std::swap(s1_begin[bds_id], s2_begin[bds_id]);
RandomAccessIterator w_begin = left_begin + bds_id * block_length;
if (right_begin < right_end) {
RandomAccessIterator q1_begin = right_begin;
right_begin = std::lower_bound(right_begin, right_end, *w_begin, compare);
std::size_t q1_length = right_begin - q1_begin;
if (enough_buffer) {
SwapBlocks(buffer_begin, w_begin, block_length);
SwapBlocks(w_begin, q1_begin, q1_length);
right_begin = w_begin;
w_begin += q1_length;
InternalBufferMerge(compare, buffer_begin, buffer_begin + block_length, q1_begin + q1_length, right_end, w_begin);
}
else {
RotateBlocks(w_begin, w_begin + block_length, q1_begin, q1_begin + q1_length);
right_begin = w_begin;
w_begin += q1_length;
SmallBufferRotationMerge(compare, w_begin, block_length, q1_begin + q1_length, right_end - (q1_begin + q1_length), buffer_begin, buffer_length);
}
}
right_end = w_begin;
}
}
return right_end;
}

template <class RandomAccessIterator, class Comparator>
inline void InPlaceMerge(Comparator compare, RandomAccessIterator left_begin, std::size_t left_length, RandomAccessIterator right_begin, std::size_t right_length, RandomAccessIterator buffer_begin, std::size_t buffer_length) {
std::size_t max_block_length = (std::size_t)sqrt(left_length);
std::size_t block_length = buffer_length;
bool enough_buffer = (buffer_length >= max_block_length);
if (!enough_buffer) block_length = left_length / buffer_length;
std::size_t bds_length = left_length / block_length + right_length / block_length;

RandomAccessIterator s1_begin = left_begin, s2_begin = left_begin + bds_length, left_end = left_begin + left_length, right_end = right_begin + right_length, array_end = right_end;
while (s2_begin < left_end && !(compare(s2_begin[-1], *s2_begin))) ++s2_begin;

if (s2_begin + bds_length >= left_end) {
SmallBufferRotationMerge(compare, left_begin, left_length, right_begin, right_length, buffer_begin, buffer_length);
return;
}

left_begin = s2_begin + bds_length;
left_length = left_end - left_begin;
InsertionSort(compare, buffer_begin, buffer_begin + left_length / block_length);

RandomAccessIterator w1_begin = left_begin;
std::size_t w1_length = left_length % block_length;
left_begin += w1_length;
left_length -= w1_length;

std::size_t bds_id = RearrangeBlocks(compare, s1_begin, s2_begin, left_begin, left_length / block_length, right_begin, right_length / block_length, buffer_begin, block_length);
right_end = MergeBlocks(compare, s1_begin, s2_begin, left_begin, right_end, buffer_begin, buffer_length, bds_id, block_length, enough_buffer);
right_begin = left_begin;

if (right_begin < right_end) {
RandomAccessIterator v1_begin = right_begin;
right_begin = std::lower_bound(right_begin, right_end, s2_begin[bds_length - 1], compare);
std::size_t v1_length = right_begin - v1_begin;
RotateBlocks(w1_begin, w1_begin + w1_length, v1_begin, right_begin);
v1_begin = w1_begin;
w1_begin += v1_length;
if (right_begin < right_end) {
if (w1_length < buffer_length) {
SwapBlocks(buffer_begin, w1_begin, w1_length);
InternalBufferMerge(compare, buffer_begin, buffer_begin + w1_length, right_begin, right_end, w1_begin);
}
else {
SmallBufferRotationMerge(compare, w1_begin, w1_length, right_begin, right_end - right_begin, buffer_begin, buffer_length);
}
}

right_begin = v1_begin;
right_end = right_begin + v1_length;

if (right_begin < right_end) {
RandomAccessIterator v1_begin = right_begin;
right_begin = std::lower_bound(right_begin, right_end, s1_begin[bds_length - 1], compare);
std::size_t v1_length = right_begin - v1_begin;
RotateBlocks(s1_begin + bds_length, v1_begin, v1_begin, right_begin);
s2_begin += v1_length;
SmallBufferRotationMerge(compare, s1_begin, bds_length, s1_begin + bds_length, v1_length, buffer_begin, buffer_length);
SmallBufferRotationMerge(compare, s2_begin, right_begin - s2_begin, right_begin, right_end - right_begin, buffer_begin, buffer_length);
}
}
}

template <class RandomAccessIterator, class Comparator>
inline void BlockRotationMerge(Comparator compare, RandomAccessIterator left_begin, std::size_t left_length, RandomAccessIterator right_begin, std::size_t right_length, std::size_t block_length) {
while (block_length < left_length) {
RandomAccessIterator breakpoint = std::lower_bound(right_begin, right_begin + right_length, left_begin[block_length - 1], compare);
std::size_t length = breakpoint - right_begin;
if (length) {
RotateBlocks(left_begin + block_length, left_begin + left_length, right_begin, breakpoint);
RotationMerge(compare, left_begin, block_length, left_begin + block_length, length);
left_begin += length;
right_begin = breakpoint;
right_length -= length;
}
left_begin += block_length;
left_length -= block_length;
}
RotationMerge(compare, left_begin, left_length, right_begin, right_length);
}

template <class RandomAccessIterator, class Comparator>
inline void InPlaceMerge(Comparator compare, RandomAccessIterator left_begin, std::size_t left_length, RandomAccessIterator right_begin, std::size_t right_length) {
std::size_t max_block_length = (std::size_t)sqrt(left_length);
std::size_t bds_length = left_length / max_block_length + right_length / max_block_length;

RandomAccessIterator s1_begin = left_begin, s2_begin = left_begin + bds_length, left_end = left_begin + left_length, right_end = right_begin + right_length, array_end = right_end;
while (s2_begin < left_end && !(compare(s2_begin[-1], *s2_begin))) ++s2_begin;

if (s2_begin + bds_length >= left_end) {
BlockRotationMerge(compare, left_begin, left_length, right_begin, right_length, max_block_length);
return;
}

left_begin = s2_begin + bds_length;

std::size_t buffer_length = ExtractBuffer(compare, left_begin, left_end, max_block_length), block_length = buffer_length;
bool enough_buffer = (buffer_length >= max_block_length);
if (!enough_buffer) block_length = left_length / buffer_length;
RandomAccessIterator buffer_begin = left_begin;
left_begin += buffer_length;
left_length = left_end - left_begin;

RandomAccessIterator w1_begin = left_begin;
std::size_t w1_length = left_length % block_length;
left_begin += w1_length;
left_length -= w1_length;

std::size_t bds_id = RearrangeBlocks(compare, s1_begin, s2_begin, left_begin, left_length / block_length, right_begin, right_length / block_length, buffer_begin, block_length);
right_end = MergeBlocks(compare, s1_begin, s2_begin, left_begin, right_end, buffer_begin, buffer_length, bds_id, block_length, enough_buffer);
right_begin = left_begin;

if (right_begin < right_end) {
RandomAccessIterator v1_begin = right_begin;
right_begin = std::lower_bound(right_begin, right_end, s2_begin[bds_length - 1], compare);
std::size_t v1_length = right_begin - v1_begin;
RotateBlocks(buffer_begin, w1_begin + w1_length, v1_begin, right_begin);
w1_begin += v1_length;
v1_begin = buffer_begin;
buffer_begin += v1_length;
if (right_begin < right_end) {
if (w1_length < buffer_length) {
SwapBlocks(buffer_begin, w1_begin, w1_length);
InternalBufferMerge(compare, buffer_begin, buffer_begin + w1_length, right_begin, right_end, w1_begin);
}
else {
SmallBufferRotationMerge(compare, w1_begin, w1_length, right_begin, right_end - right_begin, buffer_begin, buffer_length);
}
}

right_begin = v1_begin;
right_end = right_begin + v1_length;

if (right_begin < right_end) {
RandomAccessIterator v1_begin = right_begin;
right_begin = std::lower_bound(right_begin, right_end, s1_begin[bds_length - 1], compare);
std::size_t v1_length = right_begin - v1_begin;
RotateBlocks(s1_begin + bds_length, v1_begin, v1_begin, right_begin);
s2_begin += v1_length;
SmallBufferRotationMerge(compare, s1_begin, bds_length, s1_begin + bds_length, v1_length, buffer_begin, buffer_length);
SmallBufferRotationMerge(compare, s2_begin, right_begin - s2_begin, right_begin, right_end - right_begin, buffer_begin, buffer_length);
}
}

InsertionSort(compare, buffer_begin, buffer_begin + buffer_length);
RotationMerge(compare, buffer_begin, buffer_length, w1_begin, array_end - w1_begin);
}

template <class RandomAccessIterator, class Comparator>
inline void RotationMergeBack(Comparator compare, RandomAccessIterator left_begin, std::size_t left_length, RandomAccessIterator right_begin, std::size_t right_length) {
while (left_length && right_length) {
RandomAccessIterator breakpoint = std::upper_bound(left_begin, left_begin + left_length, right_begin[right_length - 1], compare);
std::size_t length = (left_begin + left_length) - breakpoint;
if (length) {
RotateBlocks(breakpoint, right_begin, right_begin, right_begin + right_length);
left_length -= length;
right_begin = breakpoint;
}
--right_length;
}
}

template <class RandomAccessIterator, class Comparator>
inline void InPlaceMergeRight(Comparator compare, RandomAccessIterator left_begin, std::size_t left_length, RandomAccessIterator right_begin, std::size_t right_length) {
RandomAccessIterator array_begin = left_begin;

std::size_t max_block_length = (std::size_t)sqrt(right_length);
std::size_t bds_length = left_length / max_block_length + right_length / max_block_length;

RandomAccessIterator s1_begin = left_begin, s2_begin = left_begin + bds_length, left_end = left_begin + left_length, right_end = right_begin + right_length;
while (s2_begin < left_end && !(compare(s2_begin[-1], *s2_begin))) ++s2_begin;

if (s2_begin + bds_length >= left_end) {
BlockRotationMerge(compare, left_begin, left_length, right_begin, right_length, max_block_length);
return;
}

left_begin = s2_begin + bds_length;
left_length = left_end - left_begin;

std::size_t buffer_length = ExtractBufferBack(compare, right_begin, right_end, max_block_length), block_length = buffer_length;
bool enough_buffer = (buffer_length >= max_block_length);
if (!enough_buffer) block_length = right_length / buffer_length;
RandomAccessIterator buffer_begin = right_end - buffer_length;
right_end -= buffer_length;
right_length -= buffer_length;

RandomAccessIterator w1_begin = left_begin;
std::size_t w1_length = left_length % block_length;
left_begin += w1_length;
left_length -= w1_length;

std::size_t bds_id = RearrangeBlocks(compare, s1_begin, s2_begin, left_begin, left_length / block_length, right_begin, right_length / block_length, buffer_begin, block_length);
right_end = MergeBlocks(compare, s1_begin, s2_begin, left_begin, right_end, buffer_begin, buffer_length, bds_id, block_length, enough_buffer);
right_begin = left_begin;

if (right_begin < right_end) {
RandomAccessIterator v1_begin = right_begin;
right_begin = std::lower_bound(right_begin, right_end, s2_begin[bds_length - 1], compare);
std::size_t v1_length = right_begin - v1_begin;
RotateBlocks(w1_begin, w1_begin + w1_length, v1_begin, right_begin);
v1_begin = w1_begin;
w1_begin += v1_length;
if (right_begin < right_end) {
if (w1_length < buffer_length) {
SwapBlocks(buffer_begin, w1_begin, w1_length);
InternalBufferMerge(compare, buffer_begin, buffer_begin + w1_length, right_begin, right_end, w1_begin);
}
else {
SmallBufferRotationMerge(compare, w1_begin, w1_length, right_begin, right_end - right_begin, buffer_begin, buffer_length);
}
}

right_begin = v1_begin;
right_end = right_begin + v1_length;

if (right_begin < right_end) {
RandomAccessIterator v1_begin = right_begin;
right_begin = std::lower_bound(right_begin, right_end, s1_begin[bds_length - 1], compare);
std::size_t v1_length = right_begin - v1_begin;
RotateBlocks(s1_begin + bds_length, v1_begin, v1_begin, right_begin);
s2_begin += v1_length;
SmallBufferRotationMerge(compare, s1_begin, bds_length, s1_begin + bds_length, v1_length, buffer_begin, buffer_length);
SmallBufferRotationMerge(compare, s2_begin, right_begin - s2_begin, right_begin, right_end - right_begin, buffer_begin, buffer_length);
}
}

InsertionSort(compare, buffer_begin, buffer_begin + buffer_length);
RotationMergeBack(compare, array_begin, buffer_begin - array_begin, buffer_begin, buffer_length);
}

template <class RandomAccessIterator, class Comparator>
inline void ParallelInPlaceMerge(Comparator compare, RandomAccessIterator left_begin, std::size_t left_length, RandomAccessIterator right_begin, std::size_t right_length, int tasks) {
if (tasks == 1 || left_length < 2 || right_length < 2) {
if (right_length > left_length) {
InPlaceMergeRight(compare, left_begin, left_length, right_begin, right_length);
}
else {
InPlaceMerge(compare, left_begin, left_length, right_begin, right_length);
}
}
else {
RandomAccessIterator left_middle, right_middle;
SplitEvenly(compare, left_begin, left_length, right_begin, right_length, left_middle, right_middle);
RotateBlocks(left_middle, left_begin + left_length, right_begin, right_middle);
#pragma omp task
ParallelInPlaceMerge(compare, left_begin, left_middle - left_begin, left_middle, right_middle - right_begin, tasks / 2);
ParallelInPlaceMerge(compare, left_middle + (right_middle - right_begin), left_length - (left_middle - left_begin), right_middle, right_length - (right_middle - right_begin), tasks / 2);
#pragma omp taskwait                    
}
}

}  
using namespace inplace_merge;

namespace merge {

template <class RandomAccessIteratorA, class RandomAccessIteratorB, class RandomAccessIteratorC, class Comparator>
inline void OutPlaceMerge(Comparator compare, RandomAccessIteratorA left_begin, RandomAccessIteratorA left_end, RandomAccessIteratorB right_begin, RandomAccessIteratorB right_end, RandomAccessIteratorC buffer) {
while (left_begin < left_end && right_begin < right_end) {
if (compare(*right_begin, *left_begin)) *buffer++ = std::move(*right_begin++);
else *buffer++ = std::move(*left_begin++);
}
if (&*left_begin != &*buffer) while (left_begin < left_end) *buffer++ = std::move(*left_begin++);
if (&*right_begin != &*buffer) while (right_begin < right_end) *buffer++ = std::move(*right_begin++);
}

template <class RandomAccessIteratorA, class RandomAccessIteratorB, class RandomAccessIteratorC, class Comparator>
inline void OutPlaceMergeBack(Comparator compare, RandomAccessIteratorA left_begin, RandomAccessIteratorA left_end, RandomAccessIteratorB right_begin, RandomAccessIteratorB right_end, RandomAccessIteratorC buffer) {
while (left_begin < left_end && right_begin < right_end) {
if (compare(*right_end, *left_end)) *buffer-- = std::move(*left_end--);
else *buffer-- = std::move(*right_end--);
}
if (&*left_end != &*buffer) while (left_begin < left_end) *buffer-- = std::move(*left_end--);
if (&*right_end != &*buffer) while (right_begin < right_end) *buffer-- = std::move(*right_end--);
}

template <class RandomAccessIteratorA, class RandomAccessIteratorB, class RandomAccessIteratorC, class Comparator>
inline void ParallelOutPlaceMerge(Comparator compare, RandomAccessIteratorA left_begin, RandomAccessIteratorA left_end, RandomAccessIteratorB right_begin, RandomAccessIteratorB right_end, RandomAccessIteratorC buffer, int tasks) {
if (tasks == 1) {
OutPlaceMerge(compare, left_begin, left_end, right_begin, right_end, buffer);
}
else {
std::size_t left_length = left_end - left_begin, right_length = right_end - right_begin;
RandomAccessIteratorA left_middle;
RandomAccessIteratorB right_middle;
SplitEvenly(compare, left_begin, left_length, right_begin, right_length, left_middle, right_middle);
#pragma omp task
ParallelOutPlaceMerge(compare, left_begin, left_middle, right_begin, right_middle, buffer, tasks / 2);
ParallelOutPlaceMerge(compare, left_middle, left_end, right_middle, right_end, buffer + (left_middle - left_begin) + (right_middle - right_begin), tasks / 2);
#pragma omp taskwait
}
}

}  
using namespace merge;

namespace hybrid_merge {

template <class RandomAccessIterator, class SortedTypePointer, class Comparator>
inline void ExternalSmallBufferRotationMerge(Comparator compare, RandomAccessIterator left_begin, std::size_t left_length, RandomAccessIterator right_begin, std::size_t right_length, SortedTypePointer buffer_begin, std::size_t buffer_length) {
while (buffer_length < left_length) {
RandomAccessIterator breakpoint = std::lower_bound(right_begin, right_begin + right_length, left_begin[buffer_length - 1], compare);
std::size_t length = breakpoint - right_begin;
if (length) {
RotateBlocks(left_begin + buffer_length, left_begin + left_length, right_begin, breakpoint);
MoveBlock(left_begin, buffer_begin, buffer_length);
OutPlaceMerge(compare, buffer_begin, buffer_begin + buffer_length, left_begin + buffer_length, left_begin + buffer_length + length, left_begin);
left_begin += length;
right_begin = breakpoint;
right_length -= length;
}
left_begin += buffer_length;
left_length -= buffer_length;
}
if (right_length) {
MoveBlock(left_begin, buffer_begin, left_length);
OutPlaceMerge(compare, buffer_begin, buffer_begin + left_length, right_begin, right_begin + right_length, left_begin);
}
}

template <class RandomAccessIterator, class SortedTypePointer, class Comparator>
inline RandomAccessIterator ExternalMergeBlocks(Comparator compare, RandomAccessIterator s1_begin, RandomAccessIterator s2_begin, RandomAccessIterator left_begin, RandomAccessIterator right_end, SortedTypePointer buffer_begin, std::size_t buffer_length, std::size_t bds_id, std::size_t block_length, bool enough_buffer) {
RandomAccessIterator right_begin = left_begin + bds_id * block_length;
while (bds_id--) {
if (compare(s1_begin[bds_id], s2_begin[bds_id])) {
right_begin = left_begin + bds_id * block_length;
}
else {
std::swap(s1_begin[bds_id], s2_begin[bds_id]);
RandomAccessIterator w_begin = left_begin + bds_id * block_length;
if (right_begin < right_end) {
RandomAccessIterator q1_begin = right_begin;
right_begin = std::lower_bound(right_begin, right_end, *w_begin, compare);
std::size_t q1_length = right_begin - q1_begin;
if (enough_buffer) {
MoveBlock(w_begin, buffer_begin, block_length);
MoveBlock(q1_begin, w_begin, q1_length);
right_begin = w_begin;
w_begin += q1_length;
OutPlaceMerge(compare, buffer_begin, buffer_begin + block_length, q1_begin + q1_length, right_end, w_begin);
}
else {
RotateBlocks(w_begin, w_begin + block_length, q1_begin, q1_begin + q1_length);
right_begin = w_begin;
w_begin += q1_length;
ExternalSmallBufferRotationMerge(compare, w_begin, block_length, q1_begin + q1_length, right_end - (q1_begin + q1_length), buffer_begin, buffer_length);
}
}
right_end = w_begin;
}
}
return right_end;
}

template <class RandomAccessIterator, class SortedTypePointer, class Comparator>
inline void HybridMerge(Comparator compare, RandomAccessIterator left_begin, std::size_t left_length, RandomAccessIterator right_begin, std::size_t right_length, RandomAccessIterator internal_buffer_begin, std::size_t internal_buffer_length, SortedTypePointer external_buffer_begin, std::size_t external_buffer_length) {
std::size_t block_length = external_buffer_length;
bool enough_buffer = (internal_buffer_length >= left_length / block_length);
if (!enough_buffer) block_length = left_length / internal_buffer_length;
std::size_t bds_length = left_length / block_length + right_length / block_length;

RandomAccessIterator s1_begin = left_begin, s2_begin = left_begin + bds_length, left_end = left_begin + left_length, right_end = right_begin + right_length, array_end = right_end;
while (s2_begin < left_end && !(compare(s2_begin[-1], *s2_begin))) ++s2_begin;

if (s2_begin + bds_length >= left_end) {
ExternalSmallBufferRotationMerge(compare, left_begin, left_length, right_begin, right_length, external_buffer_begin, external_buffer_length);
return;
}

left_begin = s2_begin + bds_length;
left_length = left_end - left_begin;

RandomAccessIterator w1_begin = left_begin;
std::size_t w1_length = left_length % block_length;
left_begin += w1_length;
left_length -= w1_length;

std::size_t bds_id = RearrangeBlocks(compare, s1_begin, s2_begin, left_begin, left_length / block_length, right_begin, right_length / block_length, internal_buffer_begin, block_length);
right_end = ExternalMergeBlocks(compare, s1_begin, s2_begin, left_begin, right_end, external_buffer_begin, external_buffer_length, bds_id, block_length, enough_buffer);
right_begin = left_begin;

if (right_begin < right_end) {
RandomAccessIterator v1_begin = right_begin;
right_begin = std::lower_bound(right_begin, right_end, s2_begin[bds_length - 1], compare);
std::size_t v1_length = right_begin - v1_begin;
RotateBlocks(w1_begin, w1_begin + w1_length, v1_begin, right_begin);
v1_begin = w1_begin;
w1_begin += v1_length;
if (right_begin < right_end) {
if (w1_length < external_buffer_length) {
MoveBlock(w1_begin, external_buffer_begin, w1_length);
InternalBufferMerge(compare, external_buffer_begin, external_buffer_begin + w1_length, right_begin, right_end, w1_begin);
}
else {
ExternalSmallBufferRotationMerge(compare, w1_begin, w1_length, right_begin, right_end - right_begin, external_buffer_begin, external_buffer_length);
}
}

right_begin = v1_begin;
right_end = right_begin + v1_length;

if (right_begin < right_end) {
RandomAccessIterator v1_begin = right_begin;
right_begin = std::lower_bound(right_begin, right_end, s1_begin[bds_length - 1], compare);
std::size_t v1_length = right_begin - v1_begin;
RotateBlocks(s1_begin + bds_length, v1_begin, v1_begin, right_begin);
s2_begin += v1_length;
ExternalSmallBufferRotationMerge(compare, s1_begin, bds_length, s1_begin + bds_length, v1_length, external_buffer_begin, external_buffer_length);
ExternalSmallBufferRotationMerge(compare, s2_begin, right_begin - s2_begin, right_begin, right_end - right_begin, external_buffer_begin, external_buffer_length);
}
}
}

template <class RandomAccessIterator, class SortedTypePointer, class Comparator>
inline void HybridMerge(Comparator compare, RandomAccessIterator left_begin, std::size_t left_length, RandomAccessIterator right_begin, std::size_t right_length, SortedTypePointer external_buffer_begin, std::size_t external_buffer_length) {
std::size_t max_block_length = external_buffer_length;
std::size_t bds_length = left_length / max_block_length + right_length / max_block_length;
if (bds_length == 0) bds_length = 1;

RandomAccessIterator s1_begin = left_begin, s2_begin = left_begin + bds_length, left_end = left_begin + left_length, right_end = right_begin + right_length, array_end = right_end;
while (s2_begin < left_end && !(compare(s2_begin[-1], *s2_begin))) ++s2_begin;

if (s2_begin + bds_length >= left_end) {
ExternalSmallBufferRotationMerge(compare, left_begin, left_length, right_begin, right_length, external_buffer_begin, external_buffer_length);
return;
}

left_begin = s2_begin + bds_length;

std::size_t buffer_length = ExtractBuffer(compare, left_begin, left_end, left_length / max_block_length), block_length = max_block_length;
bool enough_buffer = (buffer_length >= left_length / max_block_length);
if (!enough_buffer) block_length = left_length / buffer_length;
RandomAccessIterator buffer_begin = left_begin;
left_begin += buffer_length;
left_length = left_end - left_begin;

RandomAccessIterator w1_begin = left_begin;
std::size_t w1_length = left_length % block_length;
left_begin += w1_length;
left_length -= w1_length;

std::size_t bds_id = RearrangeBlocks(compare, s1_begin, s2_begin, left_begin, left_length / block_length, right_begin, right_length / block_length, buffer_begin, block_length);
right_end = ExternalMergeBlocks(compare, s1_begin, s2_begin, left_begin, right_end, external_buffer_begin, external_buffer_length, bds_id, block_length, enough_buffer);
right_begin = left_begin;

if (right_begin < right_end) {
RandomAccessIterator v1_begin = right_begin;
right_begin = std::lower_bound(right_begin, right_end, s2_begin[bds_length - 1], compare);
std::size_t v1_length = right_begin - v1_begin;
RotateBlocks(buffer_begin, w1_begin + w1_length, v1_begin, right_begin);
w1_begin += v1_length;
v1_begin = buffer_begin;
buffer_begin += v1_length;
if (right_begin < right_end) {
if (w1_length < external_buffer_length) {
MoveBlock(w1_begin, external_buffer_begin, w1_length);
InternalBufferMerge(compare, external_buffer_begin, external_buffer_begin + w1_length, right_begin, right_end, w1_begin);
}
else {
ExternalSmallBufferRotationMerge(compare, w1_begin, w1_length, right_begin, right_end - right_begin, external_buffer_begin, external_buffer_length);
}
}

right_begin = v1_begin;
right_end = right_begin + v1_length;

if (right_begin < right_end) {
RandomAccessIterator v1_begin = right_begin;
right_begin = std::lower_bound(right_begin, right_end, s1_begin[bds_length - 1], compare);
std::size_t v1_length = right_begin - v1_begin;
RotateBlocks(s1_begin + bds_length, v1_begin, v1_begin, right_begin);
s2_begin += v1_length;
ExternalSmallBufferRotationMerge(compare, s1_begin, bds_length, s1_begin + bds_length, v1_length, external_buffer_begin, external_buffer_length);
ExternalSmallBufferRotationMerge(compare, s2_begin, right_begin - s2_begin, right_begin, right_end - right_begin, external_buffer_begin, external_buffer_length);
}
}

RotationMerge(compare, buffer_begin, buffer_length, w1_begin, array_end - w1_begin);
}

template <class RandomAccessIterator, class SortedTypePointer, class Comparator>
inline void HybridMergeRight(Comparator compare, RandomAccessIterator left_begin, std::size_t left_length, RandomAccessIterator right_begin, std::size_t right_length, SortedTypePointer external_buffer_begin, std::size_t external_buffer_length) {
RandomAccessIterator array_begin = left_begin;

std::size_t max_block_length = external_buffer_length;
std::size_t bds_length = left_length / max_block_length + right_length / max_block_length;
if (bds_length == 0) bds_length = 1;

RandomAccessIterator s1_begin = left_begin, s2_begin = left_begin + bds_length, left_end = left_begin + left_length, right_end = right_begin + right_length;
while (s2_begin < left_end && !(compare(s2_begin[-1], *s2_begin))) ++s2_begin;

if (s2_begin + bds_length >= left_end) {
ExternalSmallBufferRotationMerge(compare, left_begin, left_length, right_begin, right_length, external_buffer_begin, external_buffer_length);
return;
}

left_begin = s2_begin + bds_length;
left_length = left_end - left_begin;

std::size_t buffer_length = ExtractBufferBack(compare, right_begin, right_end, right_length / max_block_length), block_length = max_block_length;
bool enough_buffer = (buffer_length >= right_length / max_block_length);
if (!enough_buffer) block_length = right_length / buffer_length;
RandomAccessIterator buffer_begin = right_end - buffer_length;
right_end -= buffer_length;
right_length -= buffer_length;

RandomAccessIterator w1_begin = left_begin;
std::size_t w1_length = left_length % block_length;
left_begin += w1_length;
left_length -= w1_length;

std::size_t bds_id = RearrangeBlocks(compare, s1_begin, s2_begin, left_begin, left_length / block_length, right_begin, right_length / block_length, buffer_begin, block_length);
right_end = ExternalMergeBlocks(compare, s1_begin, s2_begin, left_begin, right_end, external_buffer_begin, external_buffer_length, bds_id, block_length, enough_buffer);
right_begin = left_begin;

if (right_begin < right_end) {
RandomAccessIterator v1_begin = right_begin;
right_begin = std::lower_bound(right_begin, right_end, s2_begin[bds_length - 1], compare);
std::size_t v1_length = right_begin - v1_begin;
RotateBlocks(w1_begin, w1_begin + w1_length, v1_begin, right_begin);
v1_begin = w1_begin;
w1_begin += v1_length;
if (right_begin < right_end) {
if (w1_length < external_buffer_length) {
MoveBlock(w1_begin, external_buffer_begin, w1_length);
InternalBufferMerge(compare, external_buffer_begin, external_buffer_begin + w1_length, right_begin, right_end, w1_begin);
}
else {
ExternalSmallBufferRotationMerge(compare, w1_begin, w1_length, right_begin, right_end - right_begin, external_buffer_begin, external_buffer_length);
}
}

right_begin = v1_begin;
right_end = right_begin + v1_length;

if (right_begin < right_end) {
RandomAccessIterator v1_begin = right_begin;
right_begin = std::lower_bound(right_begin, right_end, s1_begin[bds_length - 1], compare);
std::size_t v1_length = right_begin - v1_begin;
RotateBlocks(s1_begin + bds_length, v1_begin, v1_begin, right_begin);
s2_begin += v1_length;
ExternalSmallBufferRotationMerge(compare, s1_begin, bds_length, s1_begin + bds_length, v1_length, external_buffer_begin, external_buffer_length);
ExternalSmallBufferRotationMerge(compare, s2_begin, right_begin - s2_begin, right_begin, right_end - right_begin, external_buffer_begin, external_buffer_length);
}
}

RotationMergeBack(compare, array_begin, buffer_begin - array_begin, buffer_begin, buffer_length);
}

template <class RandomAccessIterator, class SortedTypePointer, class Comparator>
inline void ParallelHybridMerge(Comparator compare, RandomAccessIterator left_begin, std::size_t left_length, RandomAccessIterator right_begin, std::size_t right_length, SortedTypePointer external_buffer_begin, std::size_t buffer_per_thread, int threads_with_buffer, int tasks) {
if (tasks == 1 || left_length < 2 || right_length < 2) {
int thread_num = omp_get_thread_num();
if (thread_num < threads_with_buffer) {
if (right_length > left_length) {
HybridMergeRight(compare, left_begin, left_length, right_begin, right_length, external_buffer_begin + thread_num * buffer_per_thread, buffer_per_thread);
}
else {
HybridMerge(compare, left_begin, left_length, right_begin, right_length, external_buffer_begin + thread_num * buffer_per_thread, buffer_per_thread);
}
}
else {
if (right_length > left_length) {
InPlaceMergeRight(compare, left_begin, left_length, right_begin, right_length);
}
else {
InPlaceMerge(compare, left_begin, left_length, right_begin, right_length);
}
}
}
else {
RandomAccessIterator left_middle, right_middle;
SplitEvenly(compare, left_begin, left_length, right_begin, right_length, left_middle, right_middle);
RotateBlocks(left_middle, left_begin + left_length, right_begin, right_middle);
#pragma omp task
ParallelHybridMerge(compare, left_begin, left_middle - left_begin, left_middle, right_middle - right_begin, external_buffer_begin, buffer_per_thread, threads_with_buffer, tasks / 2);
ParallelHybridMerge(compare, left_middle + (right_middle - right_begin), left_length - (left_middle - left_begin), right_middle, right_length - (right_middle - right_begin), external_buffer_begin, buffer_per_thread, threads_with_buffer, tasks / 2);
#pragma omp taskwait                    
}
}

}  
using namespace hybrid_merge;

template <class RandomAccessIteratorS, class RandomAccessIteratorD, class Comparator>
void OutPlaceMergeSort(Comparator compare, RandomAccessIteratorS begin, RandomAccessIteratorD dest, std::size_t length) {
if (length > INSERTION_SORT_MAXLEN * 2) {
std::size_t middle = length / 2;
OutPlaceMergeSort(compare, begin + middle, dest + middle, length - middle);
OutPlaceMergeSort(compare, begin, begin + length - middle, middle);
OutPlaceMerge(compare, begin + length - middle, begin + length, dest + middle, dest + length, dest);
}
else {
InsertionSort(compare, begin, begin + length / 2);
InsertionSort(compare, begin + length / 2, begin + length);
OutPlaceMerge(compare, begin, begin + length / 2, begin + length / 2, begin + length, dest);
}
}

template <class RandomAccessIterator, class SortedTypePointer, class Comparator>
void TopOutPlaceMergeSort(Comparator compare, RandomAccessIterator begin, SortedTypePointer buffer, std::size_t length) {
if (length > INSERTION_SORT_MAXLEN) {
std::size_t middle = length / 2;
OutPlaceMergeSort(compare, begin + middle, buffer, length - middle);
OutPlaceMergeSort(compare, begin, begin + length - middle, middle);
OutPlaceMerge(compare, begin + length - middle, begin + length, buffer, buffer + length - middle, begin);
}
else {
InsertionSort(compare, begin, begin + length);
}
}

template <class RandomAccessIterator, class SortedTypePointer, class Comparator>
void ParallelOutPlaceMergeSort(Comparator compare, RandomAccessIterator begin, SortedTypePointer buffer, std::size_t length, int tasks, bool swapped = false) {
if (tasks == 1) {
if (swapped) OutPlaceMergeSort(compare, buffer, begin, length);
else TopOutPlaceMergeSort(compare, begin, buffer, length);
}
else if (length > INSERTION_SORT_MAXLEN || swapped) {
std::size_t middle = length / 2;
#pragma omp task
ParallelOutPlaceMergeSort(compare, buffer, begin, middle, tasks / 2, !swapped);
ParallelOutPlaceMergeSort(compare, buffer + middle, begin + middle, length - middle, tasks / 2, !swapped);
#pragma omp taskwait
ParallelOutPlaceMerge(compare, buffer, buffer + middle, buffer + middle, buffer + length, begin, tasks);
}
else {
InsertionSort(compare, begin, begin + length);
}
}

template <class RandomAccessIterator, class Comparator>
inline void OutPlaceMergeSort(Comparator compare, RandomAccessIterator begin, RandomAccessIterator end, int threads) {
typedef typename std::remove_reference<decltype(*begin)>::type SortedType;
std::size_t length = end - begin;
SortedType* buffer = new SortedType[length];
#pragma omp parallel num_threads(threads)
#pragma omp master
ParallelOutPlaceMergeSort(compare, begin, buffer, length, upper_power_of_two(threads * K_THREADS_TASK));
delete[] buffer;
}

template <class RandomAccessIterator, class Comparator>
void InPlaceMergeSort(Comparator compare, RandomAccessIterator begin, std::size_t length, RandomAccessIterator buffer_begin, std::size_t buffer_length) {
if (length > IN_PLACE_INSERTION_SORT_MAXLEN) {
std::size_t middle = length / 2;
InPlaceMergeSort(compare, begin, middle, buffer_begin, buffer_length);
InPlaceMergeSort(compare, begin + middle, length - middle, buffer_begin, buffer_length);
if (middle <= buffer_length) {
SwapBlocks(begin, buffer_begin, middle);
InternalBufferMerge(compare, buffer_begin, buffer_begin + middle, begin + middle, begin + length, begin);
}
else {
InPlaceMerge(compare, begin, middle, begin + middle, length - middle, buffer_begin, buffer_length);
}
}
else {
InsertionSort(compare, begin, begin + length);
}
}

template <class RandomAccessIterator, class Comparator>
inline void InPlaceMergeSortTask(Comparator compare, RandomAccessIterator begin, RandomAccessIterator end) {
std::size_t buffer_length = ExtractBufferAssorted(compare, begin, end, (std::size_t)sqrt((end - begin)));
InPlaceMergeSort(compare, begin + buffer_length, end - begin - buffer_length, begin, buffer_length);
InsertionSort(compare, begin, begin + buffer_length);
RotationMerge(compare, begin, buffer_length, begin + buffer_length, end - begin - buffer_length);
}

template <class RandomAccessIterator, class Comparator>
void ParallelInPlaceMergeSort(Comparator compare, RandomAccessIterator begin, std::size_t length, int tasks) {
if (tasks == 1) InPlaceMergeSortTask(compare, begin, begin + length);
else if (length > IN_PLACE_INSERTION_SORT_MAXLEN) {
std::size_t middle = length / 2;
#pragma omp task
ParallelInPlaceMergeSort(compare, begin, middle, tasks / 2);
ParallelInPlaceMergeSort(compare, begin + middle, length - middle, tasks / 2);
#pragma omp taskwait
ParallelInPlaceMerge(compare, begin, middle, begin + middle, length - middle, tasks);
}
else {
InsertionSort(compare, begin, begin + length);
}
}

template <class RandomAccessIterator, class Comparator>
inline void InPlaceMergeSort(Comparator compare, RandomAccessIterator begin, RandomAccessIterator end, int threads = omp_get_max_threads()) {
#pragma omp parallel num_threads(threads)
#pragma omp master
ParallelInPlaceMergeSort(compare, begin, end - begin, upper_power_of_two(threads * INPLACE_K_THREADS_TASK));
}

template <class RandomAccessIterator, class SortedTypePointer, class Comparator>
void HybridSort(Comparator compare, RandomAccessIterator begin, std::size_t length, RandomAccessIterator internal_buffer_begin, std::size_t internal_buffer_length, SortedTypePointer external_buffer_begin, std::size_t external_buffer_length) {
if (length > INSERTION_SORT_MAXLEN) {
std::size_t middle = length / 2;
if (external_buffer_length < length - middle) {
HybridSort(compare, begin, middle, internal_buffer_begin, internal_buffer_length, external_buffer_begin, external_buffer_length);
HybridSort(compare, begin + middle, length - middle, internal_buffer_begin, internal_buffer_length, external_buffer_begin, external_buffer_length);
HybridMerge(compare, begin, middle, begin + middle, length - middle, internal_buffer_begin, internal_buffer_length, external_buffer_begin, external_buffer_length);
}
else {
TopOutPlaceMergeSort(compare, begin, external_buffer_begin, length);
}
}
else {
InsertionSort(compare, begin, begin + length);
}
}

template <class RandomAccessIterator, class SortedTypePointer, class Comparator>
inline void HybridSortTask(Comparator compare, RandomAccessIterator begin, std::size_t length, SortedTypePointer external_buffer_begin, std::size_t external_buffer_length) {
std::size_t internal_buffer_length = (length / 2) / external_buffer_length, max_length = (std::size_t)sqrt(length);
if (internal_buffer_length > max_length) internal_buffer_length = max_length;
internal_buffer_length = ExtractBufferAssorted(compare, begin, begin + length, internal_buffer_length);
RandomAccessIterator internal_buffer_begin = begin;
begin += internal_buffer_length;
length -= internal_buffer_length;

HybridSort(compare, begin, length, internal_buffer_begin, internal_buffer_length, external_buffer_begin, external_buffer_length);

RotationMerge(compare, internal_buffer_begin, internal_buffer_length, begin, length);
}

template <class RandomAccessIterator, class SortedTypePointer, class Comparator>
inline void ParallelHybridSort(Comparator compare, RandomAccessIterator begin, std::size_t length, SortedTypePointer external_buffer_begin, std::size_t buffer_per_thread, int threads_with_buffer, int tasks) {
if (tasks == 1) {
int thread_num = omp_get_thread_num();
if (thread_num < threads_with_buffer) {
HybridSortTask(compare, begin, length, external_buffer_begin + thread_num * buffer_per_thread, buffer_per_thread);
}
else {
InPlaceMergeSortTask(compare, begin, begin + length);
}
}
else if (length > IN_PLACE_INSERTION_SORT_MAXLEN) {
std::size_t middle = length / 2;
#pragma omp task
ParallelHybridSort(compare, begin, middle, external_buffer_begin, buffer_per_thread, threads_with_buffer, tasks / 2);
ParallelHybridSort(compare, begin + middle, length - middle, external_buffer_begin, buffer_per_thread, threads_with_buffer, tasks / 2);
#pragma omp taskwait
ParallelHybridMerge(compare, begin, middle, begin + middle, length - middle, external_buffer_begin, buffer_per_thread, threads_with_buffer, tasks);
}
else {
InsertionSort(compare, begin, begin + length);
}
}

template <class RandomAccessIterator, class Comparator>
inline void HybridSort(Comparator compare, RandomAccessIterator begin, RandomAccessIterator end, std::size_t external_buffer_length, int threads) {
std::size_t max_length;
if (external_buffer_length >= (std::size_t)(end - begin)) {
OutPlaceMergeSort(compare, begin, end, threads);
}
else if (external_buffer_length < (max_length = (std::size_t)sqrt(2 * (end - begin) / upper_power_of_two(threads * INPLACE_K_THREADS_TASK)))) {
InPlaceMergeSort(compare, begin, end, threads);
}
else {
int threads_with_buffer = threads;
std::size_t buffer_per_thread = external_buffer_length / threads_with_buffer;
if (buffer_per_thread < max_length) {
threads_with_buffer = external_buffer_length / max_length;
buffer_per_thread = external_buffer_length / threads_with_buffer;
}

typedef typename std::remove_reference<decltype(*begin)>::type SortedType;
SortedType* external_buffer_begin = new SortedType[threads_with_buffer * buffer_per_thread];

#pragma omp parallel num_threads(threads)
#pragma omp master  
ParallelHybridSort(compare, begin, end - begin, external_buffer_begin, buffer_per_thread, threads_with_buffer, upper_power_of_two(threads * INPLACE_K_THREADS_TASK));

delete[] external_buffer_begin;
}
}
namespace serial {
template <class RandomAccessIterator, class Comparator>
inline void OutPlaceMergeSort(Comparator compare, RandomAccessIterator begin, RandomAccessIterator end) {
typedef typename std::remove_reference<decltype(*begin)>::type SortedType;
std::size_t length = (end - begin + 1) / 2;
SortedType* buffer = new SortedType[length];
TopOutPlaceMergeSort(compare, begin, buffer, end - begin);
delete[] buffer;
}

template <class RandomAccessIterator, class SortedTypePointer, class Comparator>
void HybridSort(Comparator compare, RandomAccessIterator begin, std::size_t length, RandomAccessIterator internal_buffer_begin, std::size_t internal_buffer_length, SortedTypePointer external_buffer_begin, std::size_t external_buffer_length) {
if (length > INSERTION_SORT_MAXLEN) {
std::size_t middle = length / 2;
if (external_buffer_length < length - middle) {
HybridSort(compare, begin, middle, internal_buffer_begin, internal_buffer_length, external_buffer_begin, external_buffer_length);
HybridSort(compare, begin + middle, length - middle, internal_buffer_begin, internal_buffer_length, external_buffer_begin, external_buffer_length);
HybridMerge(compare, begin, middle, begin + middle, length - middle, internal_buffer_begin, internal_buffer_length, external_buffer_begin, external_buffer_length);
}
else {
TopOutPlaceMergeSort(compare, begin, external_buffer_begin, length);
}
}                                                
else {
InsertionSort(compare, begin, begin + length);
}
}

template <class RandomAccessIterator, class Comparator>
inline void HybridSort(Comparator compare, RandomAccessIterator begin, RandomAccessIterator end, std::size_t external_buffer_length) {
std::size_t max_length;
if (external_buffer_length >= (std::size_t)(end - begin + 1) / 2) {
OutPlaceMergeSort(compare, begin, end);
}
else if (external_buffer_length < (max_length = (std::size_t)(sqrt(end - begin))) / 16 || external_buffer_length == 0) {
InPlaceMergeSort(compare, begin, end);
}
else {
typedef typename std::remove_reference<decltype(*begin)>::type SortedType;
SortedType* external_buffer_begin = new SortedType[external_buffer_length];

std::size_t internal_buffer_length = ((end - begin) / 2) / external_buffer_length;
if (internal_buffer_length > max_length) internal_buffer_length = max_length;
internal_buffer_length = ExtractBufferAssorted(compare, begin, end, internal_buffer_length);
RandomAccessIterator internal_buffer_begin = begin;
begin += internal_buffer_length;

HybridSort(compare, begin, end - begin, internal_buffer_begin, internal_buffer_length, external_buffer_begin, external_buffer_length);

RotationMerge(compare, internal_buffer_begin, internal_buffer_length, begin, end - begin);
delete[] external_buffer_begin;
}
}
}  

}  

template <class RandomAccessIterator, class Comparator>
void HybridSort(RandomAccessIterator begin, RandomAccessIterator end, Comparator compare, std::size_t buffer_length = -1LL, int threads = omp_get_max_threads()) {
if (buffer_length > end - begin) buffer_length = end - begin;
if (threads == 1) {
parallel_stable_sorting::serial::HybridSort(compare, begin, end, buffer_length);
}
else {
parallel_stable_sorting::HybridSort(compare, begin, end, buffer_length, threads);
}
}

template <class RandomAccessIterator>
void HybridSort(RandomAccessIterator begin, RandomAccessIterator end) {
HybridSort(begin, end, std::less<>());
}

#endif  