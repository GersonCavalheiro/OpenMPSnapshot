
#ifndef PRIMESIEVE_ITERATOR_HPP
#define PRIMESIEVE_ITERATOR_HPP

#include <stdint.h>
#include <cstddef>
#include <limits>

#if __cplusplus >= 202002L && \
defined(__has_cpp_attribute)
#if __has_cpp_attribute(unlikely)
#define IF_UNLIKELY_PRIMESIEVE(x) if (x) [[unlikely]]
#endif
#elif defined(__has_builtin)
#if __has_builtin(__builtin_expect)
#define IF_UNLIKELY_PRIMESIEVE(x) if (__builtin_expect(!!(x), 0))
#endif
#endif
#if !defined(IF_UNLIKELY_PRIMESIEVE)
#define IF_UNLIKELY_PRIMESIEVE(x) if (x)
#endif

#if defined(min) || defined(max)
#undef min
#undef max
#if __cplusplus >= 202301L
#warning "Undefining min()/max() macros. Please define NOMINMAX before including <Windows.h>"
#elif defined(_MSC_VER) || defined(__GNUG__)
#pragma message("Undefining min()/max() macros. Please define NOMINMAX before including <Windows.h>")
#endif
#endif

namespace primesieve {

struct iterator
{
iterator() noexcept;

iterator(uint64_t start, uint64_t stop_hint = std::numeric_limits<uint64_t>::max()) noexcept;

void jump_to(uint64_t start, uint64_t stop_hint = std::numeric_limits<uint64_t>::max()) noexcept;

iterator(const iterator&) = delete;
iterator& operator=(const iterator&) = delete;

iterator(iterator&&) noexcept;
iterator& operator=(iterator&&) noexcept;

~iterator();

void clear() noexcept;

void generate_next_primes();
void generate_prev_primes();

uint64_t next_prime()
{
i_ += 1;
IF_UNLIKELY_PRIMESIEVE(i_ >= size_)
generate_next_primes();
return primes_[i_];
}

uint64_t prev_prime()
{
IF_UNLIKELY_PRIMESIEVE(i_ == 0)
generate_prev_primes();
i_ -= 1;
return primes_[i_];
}

std::size_t i_;
std::size_t size_;
uint64_t start_;
uint64_t stop_hint_;
uint64_t* primes_;
void* memory_;
};

} 

#endif
