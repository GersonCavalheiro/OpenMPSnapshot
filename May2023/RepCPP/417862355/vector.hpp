#pragma once

typedef unsigned uint;

#include <iostream>
#include <stdexcept>
#include <iterator>     
#include <algorithm>


#ifndef UPSIZE_POLICY
#define UPSIZE_POLICY 1.1
#endif
#ifndef DOWNSIZE_POLICY
#define DOWNSIZE_POLICY 0.85
#endif
#ifndef MIN_VECTOR_SIZE
#define MIN_VECTOR_SIZE 1000
#endif

namespace quids::utils {

float upsize_policy = UPSIZE_POLICY;
float downsize_policy = DOWNSIZE_POLICY;
size_t min_vector_size = MIN_VECTOR_SIZE;

template <typename T>
class fast_vector {
private:
mutable T* ptr = NULL;
mutable T* unaligned_ptr = NULL;
mutable size_t size_ = 0;

public:
template<typename Int=size_t>
explicit fast_vector(const Int n = 0) {
resize(n);
}

~fast_vector() {
if (ptr != NULL) {
free(ptr);
ptr = NULL;
size_ = 0;
}
}

size_t push_back(T) {
exit(0);
return 0;
}

T pop_back() {
return ptr[size_-- - 1];
}

size_t size() const {
return size_;
}

template<typename Int=size_t>
T& operator[](Int index) {
return *(ptr + index);
}

template<typename Int=size_t>
T operator[](size_t index) const {
return *(ptr + index);
}

template<typename Int=size_t>
T& at(const Int index) {
if (index > size_) {
std::cerr << "index out of bound in fast vector !\n";
throw;
}

return *(ptr + index);
}

template<typename Int=size_t>
T at(const Int index) const {
if (index > size_) {
std::cerr << "index out of bound in fast vector !\n";
throw;
}

return *(ptr + index);
}


template<typename Int=size_t>
void resize(const Int n_, const uint align_byte_length_=std::alignment_of<T>()) const {
size_t n = std::max(min_vector_size, (size_t)n_); 

if (size_ < n || 
n*upsize_policy < size_*downsize_policy) { 
size_t old_size_ = size_;

size_ = n*upsize_policy;
int offset = std::distance(unaligned_ptr, ptr);
unaligned_ptr = (T*)realloc(unaligned_ptr, (size_ + align_byte_length_)*sizeof(T));

if (unaligned_ptr == NULL)
throw std::runtime_error("bad allocation in fast_vector !!");

ptr = unaligned_ptr + offset;
if (align_byte_length_ > 1) {
size_t allign_offset = ((size_t)ptr)%align_byte_length_;

if (allign_offset != 0)
if (allign_offset <= offset) {
std::rotate<char*>(((char*)ptr) - allign_offset, (char*)ptr, ((char*)ptr) + old_size_*sizeof(T));
ptr -= allign_offset;
} else {
allign_offset = align_byte_length_ - allign_offset;
std::rotate<char*>((char*)ptr, ((char*)ptr) + old_size_*sizeof(T), ((char*)ptr) + old_size_*sizeof(T) + allign_offset);
ptr += allign_offset;
}
}
}
}

inline T* begin() const {
return ptr;
}

inline T* end() const {
return begin() + size_;
}
};
}