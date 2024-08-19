#pragma once
#include <iterator>

using std::iterator_traits;





template <class I>
class BoundedDequeView {
protected:
const I xb, xe;
I ib, ie;
size_t n;


public:
using value_type = typename iterator_traits<I>::value_type;
protected:
using T = value_type;


public:
inline size_t size()  const { return n; }
inline bool   empty() const { return n == 0; }


public:
inline auto back()  const { return ie == xb? *(xe - 1) : *(ie - 1); }
inline auto front() const { return *ib; }


public:
inline void push_back(const T& v) {
++n; *(ie++) = v;
if (ie == xe) ie = xb;
}

inline void push_front(const T& v) {
if (ib == xb) ib = xe;
++n; *(--ib) = v;
}

inline auto pop_back() {
if (ie == xb) ie = xe;
--n; return *(--ie);
}

inline auto pop_front() {
--n; auto v = *(ib++);
if (ib == xe) ib = xb;
return v;
}


BoundedDequeView(I xb, I xe) :
xb(xb), xe(xe), ib(xb), ie(xb), n(0) {}
};


template <class I>
inline auto bounded_deque_view(I xb, I xe) {
return BoundedDequeView<I>(xb, xe);
}
template <class J>
inline auto boundedDequeView(J& x) {
return bounded_deque_view(x.begin(), x.end());
}





template <class I>
class RBoundedDequeView {
protected:
const I xb, xe;
I ib, ie;


public:
using value_type = typename iterator_traits<I>::value_type;
protected:
using T = value_type;


public:
inline bool empty() const { return ib == ie; }


public:
inline auto back()  const { return ie == xb? *(xe - 1) : *(ie - 1); }
inline auto front() const { return *ib; }


public:
inline void push_back(const T& v) {
*(ie++) = v;
if (ie == xe) ie = xb;
}

inline void push_front(const T& v) {
if (ib == xb) ib = xe;
*(--ib) = v;
}

inline auto pop_back() {
if (ie == xb) ie = xe;
return *(--ie);
}

inline auto pop_front() {
auto v = *(ib++);
if (ib == xe) ib = xb;
return v;
}


RBoundedDequeView(I xb, I xe) :
xb(xb), xe(xe), ib(xb), ie(xb) {}
};


template <class I>
inline auto rbounded_deque_view(I xb, I xe) {
return RBoundedDequeView<I>(xb, xe);
}
template <class J>
inline auto rboundedDequeView(J& x) {
return rbounded_deque_view(x.begin(), x.end());
}
