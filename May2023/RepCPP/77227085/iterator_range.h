

#pragma once

#include <utility>

namespace rawspeed {

template <typename Iter> class iterator_range {
Iter begin_iterator;
Iter end_iterator;

public:
iterator_range(Iter begin_iterator_, Iter end_iterator_)
: begin_iterator(std::move(begin_iterator_)),
end_iterator(std::move(end_iterator_)) {}

[[nodiscard]] Iter begin() const { return begin_iterator; }
[[nodiscard]] Iter end() const { return end_iterator; }
[[nodiscard]] bool empty() const { return begin_iterator == end_iterator; }
};

template <class T> iterator_range<T> make_range(T x, T y) {
return iterator_range<T>(std::move(x), std::move(y));
}

template <typename T> iterator_range<T> make_range(std::pair<T, T> p) {
return iterator_range<T>(std::move(p.first), std::move(p.second));
}

} 
