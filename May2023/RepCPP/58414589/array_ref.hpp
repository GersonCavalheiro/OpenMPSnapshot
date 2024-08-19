

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_ARRAY_REF_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_ARRAY_REF_HPP

#include <array>
#include <stddef.h>
#include <vector>
#include "compiler_macros.hpp"
#include <initializer_list>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {


template <typename T>
class array_ref {
private:
const T *ptr_ = nullptr;
size_t sz_ = 0;

public:
using value_type = T;
using const_pointer = const value_type *;
using const_iterator = const_pointer;
using const_reference = const value_type &;
using const_reverse_iterator = std::reverse_iterator<const_iterator>;
using difference_type = ptrdiff_t;
using iterator = const_pointer;
using pointer = value_type *;
using reference = value_type &;
using reverse_iterator = std::reverse_iterator<iterator>;
using size_type = size_t;

array_ref() = default;
array_ref(const T *ptr) : ptr_(ptr), sz_(1) {}
array_ref(const T *ptr, size_t size) : ptr_(ptr), sz_(size) {}

array_ref(const T *begin, const T *end) : ptr_(begin), sz_(end - begin) {}

#if SC_GNUC_VERSION_GE(9)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winit-list-lifetime"
#endif
constexpr array_ref(const std::initializer_list<T> &v)
: ptr_(v.begin() == v.end() ? nullptr : v.begin()), sz_(v.size()) {}
#if SC_GNUC_VERSION_GE(9)
#pragma GCC diagnostic pop
#endif

template <typename A>
array_ref(const std::vector<T, A> &v) : ptr_(v.data()), sz_(v.size()) {}

template <size_t N>
constexpr array_ref(const std::array<T, N> &v) : ptr_(v.data()), sz_(N) {}

template <size_t N>
constexpr array_ref(const T (&v)[N]) : ptr_(v), sz_(N) {}

bool empty() const { return sz_ == 0; }

const T *data() const { return ptr_; }

size_t size() const { return sz_; }

const T &front() const { return (*this)[0]; }

const T &back() const { return (*this)[sz_ - 1]; }

const T &operator[](size_t i) const {
assert(i < sz_);
return ptr_[i];
}

template <typename U>
typename std::enable_if<std::is_same<U, T>::value, array_ref<T>>::type &
operator=(U &&)
= delete;

template <typename U>
typename std::enable_if<std::is_same<U, T>::value, array_ref<T>>::type &
operator=(std::initializer_list<U>)
= delete;

iterator begin() const { return ptr_; }
iterator end() const { return ptr_ + sz_; }

reverse_iterator rbegin() const { return reverse_iterator(end()); }
reverse_iterator rend() const { return reverse_iterator(begin()); }

std::vector<T> as_vector() const {
return std::vector<T>(ptr_, ptr_ + sz_);
}
};
} 
} 
} 
} 

#endif
