#pragma once
#include <iterator>

using std::iterator_traits;





template <class I>
class DequeView {
protected:
using T = typename iterator_traits<I>::value_type;
const I xb, xe;
I ib, ie;
size_t n;

public:
using value_type = T;
inline size_t size()  const { return n; }
inline bool   empty() const { return n==0; }

public:
inline auto back()  const { return ie==xb? *(xe-1) : *(ie-1); }
inline auto front() const { return *ib; }

public:
inline void push_back(const T& v) {
if (ib!=ie || n==0) ++n;
*(ie++) = v;
if (ie==xe) ie = xb;
}

inline void push_front(const T& v) {
if (ib==xb) ib = xe;
*(--ib) = v;
if (ib!=ie || n==0) ++n;
}

inline auto pop_back() {
if (ie==xb) ie = xe;
if (n>0) --n;
return *(--ie);
}

inline auto pop_front() {
if (n>0) --n;
auto v = *(ib++);
if (ib==xe) ib = xb;
return v;
}

DequeView(I xb, I xe) :
xb(xb), xe(xe), ib(xb), ie(xb), n(0) {}
};

template <class I>
inline auto deque_view(I xb, I xe) {
return DequeView<I>(xb, xe);
}
template <class J>
inline auto dequeView(J& x) {
return deque_view(x.begin(), x.end());
}





template <class I>
class UnsizedDequeView {
protected:
using T = typename iterator_traits<I>::value_type;
const I xb, xe;
I ib, ie;

public:
using value_type = T;
inline bool empty() const { return ib==ie; }

public:
inline auto back()  const { return ie==xb? *(xe-1) : *(ie-1); }
inline auto front() const { return *ib; }

public:
inline void push_back(const T& v) {
*(ie++) = v;
if (ie==xe) ie = xb;
}

inline void push_front(const T& v) {
if (ib==xb) ib = xe;
*(--ib) = v;
}

inline auto pop_back() {
if (ie==xb) ie = xe;
return *(--ie);
}

inline auto pop_front() {
auto v = *(ib++);
if (ib==xe) ib = xb;
return v;
}

UnsizedDequeView(I xb, I xe) :
xb(xb), xe(xe), ib(xb), ie(xb) {}
};

template <class I>
inline auto unsized_deque_view(I xb, I xe) {
return UnsizedDequeView<I>(xb, xe);
}
template <class J>
inline auto unsizedDequeView(J& x) {
return unsized_deque_view(x.begin(), x.end());
}
