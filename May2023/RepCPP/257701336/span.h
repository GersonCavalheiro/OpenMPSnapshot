

#ifndef OPENSSL_HEADER_SSL_SPAN_H
#define OPENSSL_HEADER_SSL_SPAN_H

#include <openssl/base.h>

#if !defined(BORINGSSL_NO_CXX)

extern "C++" {

#include <algorithm>
#include <cstdlib>
#include <type_traits>

namespace bssl {

template <typename T>
class Span;

namespace internal {
template <typename T>
class SpanBase {
static_assert(std::is_const<T>::value,
"Span<T> must be derived from SpanBase<const T>");

friend bool operator==(Span<T> lhs, Span<T> rhs) {
if (lhs.size() != rhs.size()) {
return false;
}
for (T *l = lhs.begin(), *r = rhs.begin(); l != lhs.end() && r != rhs.end();
++l, ++r) {
if (*l != *r) {
return false;
}
}
return true;
}

friend bool operator!=(Span<T> lhs, Span<T> rhs) { return !(lhs == rhs); }
};
}  

template <typename T>
class Span : private internal::SpanBase<const T> {
private:
template <typename C>
using EnableIfContainer = std::enable_if<
std::is_convertible<decltype(std::declval<C>().data()), T *>::value &&
std::is_integral<decltype(std::declval<C>().size())>::value>;

static const size_t npos = static_cast<size_t>(-1);

public:
constexpr Span() : Span(nullptr, 0) {}
constexpr Span(T *ptr, size_t len) : data_(ptr), size_(len) {}

template <size_t N>
constexpr Span(T (&array)[N]) : Span(array, N) {}

template <
typename C, typename = typename EnableIfContainer<C>::type,
typename = typename std::enable_if<std::is_const<T>::value, C>::type>
Span(const C &container) : data_(container.data()), size_(container.size()) {}

template <
typename C, typename = typename EnableIfContainer<C>::type,
typename = typename std::enable_if<!std::is_const<T>::value, C>::type>
explicit Span(C &container)
: data_(container.data()), size_(container.size()) {}

T *data() const { return data_; }
size_t size() const { return size_; }
bool empty() const { return size_ == 0; }

T *begin() const { return data_; }
const T *cbegin() const { return data_; }
T *end() const { return data_ + size_; };
const T *cend() const { return end(); };

T &front() const {
if (size_ == 0) {
abort();
}
return data_[0];
}
T &back() const {
if (size_ == 0) {
abort();
}
return data_[size_ - 1];
}

T &operator[](size_t i) const {
if (i >= size_) {
abort();
}
return data_[i];
}
T &at(size_t i) const { return (*this)[i]; }

Span subspan(size_t pos = 0, size_t len = npos) const {
if (pos > size_) {
abort();  
}
return Span(data_ + pos, std::min(size_ - pos, len));
}

private:
T *data_;
size_t size_;
};

template <typename T>
const size_t Span<T>::npos;

template <typename T>
Span<T> MakeSpan(T *ptr, size_t size) {
return Span<T>(ptr, size);
}

template <typename C>
auto MakeSpan(C &c) -> decltype(MakeSpan(c.data(), c.size())) {
return MakeSpan(c.data(), c.size());
}

template <typename T>
Span<const T> MakeConstSpan(T *ptr, size_t size) {
return Span<const T>(ptr, size);
}

template <typename C>
auto MakeConstSpan(const C &c) -> decltype(MakeConstSpan(c.data(), c.size())) {
return MakeConstSpan(c.data(), c.size());
}

}  

}  

#endif  

#endif  
