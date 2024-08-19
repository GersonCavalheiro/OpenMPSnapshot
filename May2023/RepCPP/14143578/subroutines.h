#pragma once
#include "dg/topology/functions.h"
#include "dg/backend/config.h"

namespace dg{

struct IDENTITY
{
template<class T>
DG_DEVICE T operator()(T x)const{return x;}
};



struct equals
{
template< class T1, class T2>
DG_DEVICE void operator()( T1 x, T2& y) const
{
y = x;
}
};
struct plus_equals
{
template< class T1, class T2>
DG_DEVICE void operator()( T1 x, T2& y) const
{
y += x;
}
};
struct minus_equals
{
template< class T1, class T2>
DG_DEVICE void operator()( T1 x, T2& y) const
{
y -= x;
}
};
struct times_equals
{
template< class T1, class T2>
DG_DEVICE void operator()( T1 x, T2& y) const
{
y *= x;
}
};
struct divides_equals
{
template< class T1, class T2>
DG_DEVICE void operator()( T1 x, T2& y) const
{
y /= x;
}
};


struct divides
{
template< class T1, class T2>
DG_DEVICE T1 operator()( T1 x1, T2 x2) const
{
return x1/x2;
}
};

struct Sum
{
template< class T1, class ...Ts>
DG_DEVICE T1 operator()( T1 x, Ts... rest) const
{
T1 tmp = T1{0};
sum( tmp, x, rest...);
return tmp;
}
private:
template<class T, class ...Ts>
DG_DEVICE void sum( T& tmp, T x, Ts... rest) const
{
tmp += x;
sum( tmp, rest...);
}

template<class T>
DG_DEVICE void sum( T& tmp, T x) const
{
tmp += x;
}
};

struct PairSum
{
template< class T, class ...Ts>
DG_DEVICE T operator()( T a, T x, Ts... rest) const
{
T tmp = T{0};
sum( tmp, a, x, rest...);
return tmp;
}
private:
template<class T, class ...Ts>
DG_DEVICE void sum( T& tmp, T alpha, T x, Ts... rest) const
{
tmp = DG_FMA( alpha, x, tmp);
sum( tmp, rest...);
}

template<class T>
DG_DEVICE void sum( T& tmp, T alpha, T x) const
{
tmp = DG_FMA(alpha, x, tmp);
}
};
struct TripletSum
{
template< class T1, class ...Ts>
DG_DEVICE T1 operator()( T1 a, T1 x1, T1 y1, Ts... rest) const
{
T1 tmp = T1{0};
sum( tmp, a, x1, y1, rest...);
return tmp;
}
private:
template<class T, class ...Ts>
DG_DEVICE void sum( T& tmp, T alpha, T x, T y, Ts... rest) const
{
tmp = DG_FMA( alpha*x, y, tmp);
sum( tmp, rest...);
}

template<class T>
DG_DEVICE void sum( T& tmp, T alpha, T x, T y) const
{
tmp = DG_FMA(alpha*x, y, tmp);
}
};



struct EmbeddedPairSum
{
template< class T1, class ...Ts>
DG_DEVICE void operator()( T1& y, T1& yt, T1 b, T1 bt, Ts... rest) const
{
y = b*y;
yt = bt*yt;
sum( y, yt, rest...);
}
private:
template< class T1,  class ...Ts>
DG_DEVICE void sum( T1& y_1, T1& yt_1, T1 b, T1 bt, T1 k, Ts... rest) const
{
y_1 = DG_FMA( b, k, y_1);
yt_1 = DG_FMA( bt, k, yt_1);
sum( y_1, yt_1, rest...);
}

template< class T1>
DG_DEVICE void sum( T1& y_1, T1& yt_1, T1 b, T1 bt, T1 k) const
{
y_1 = DG_FMA( b, k, y_1);
yt_1 = DG_FMA( bt, k, yt_1);
}
};

template<class BinarySub, class Functor>
struct Evaluate
{
Evaluate( BinarySub sub, Functor g):
m_f( sub),
m_g( g) {}
#ifdef __CUDACC__
#pragma hd_warning_disable
#endif
template< class T, class... Ts>
DG_DEVICE void operator() ( T& y, Ts... xs){
m_f(m_g(xs...), y);
}
private:
BinarySub m_f;
Functor m_g;
};

template<class T>
struct Scal
{
Scal( T a): m_a(a){}
DG_DEVICE
void operator()( T& y)const{
y *= m_a;
}
private:
T m_a;
};

template<class T>
struct Plus
{
Plus( T a): m_a(a){}
DG_DEVICE
void operator()( T& y) const{
y += m_a;
}
private:
T m_a;
};

template<class T>
struct Axpby
{
Axpby( T a, T b): m_a(a), m_b(b){}
DG_DEVICE
void operator()( T x, T& y)const {
T temp = y*m_b;
y = DG_FMA( m_a, x, temp);
}
private:
T m_a, m_b;
};
template<class T>
struct AxyPby
{
AxyPby( T a, T b): m_a(a), m_b(b){}
DG_DEVICE
void operator()( T x, T& y)const {
T temp = y*m_b;
y = DG_FMA( m_a*x, y, temp);
}
private:
T m_a, m_b;
};

template<class T>
struct Axpbypgz
{
Axpbypgz( T a, T b, T g): m_a(a), m_b(b), m_g(g){}
DG_DEVICE
void operator()( T x, T y, T& z)const{
T temp = z*m_g;
temp = DG_FMA( m_a, x, temp);
temp = DG_FMA( m_b, y, temp);
z = temp;
}
private:
T m_a, m_b, m_g;
};

template<class T>
struct PointwiseDot
{
PointwiseDot( T a, T b, T g = (T)0): m_a(a), m_b(b), m_g(g) {}
DG_DEVICE void operator()( T x, T y, T& z)const{
T temp = z*m_b;
z = DG_FMA( m_a*x, y, temp);
}
DG_DEVICE
void operator()( T x1, T x2, T x3, T& y)const{
T temp = y*m_b;
y = DG_FMA( m_a*x1, x2*x3, temp);
}
DG_DEVICE
void operator()( T x1, T y1, T x2, T y2, T& z)const{
T temp = z*m_g;
temp = DG_FMA( m_a*x1, y1, temp);
temp = DG_FMA( m_b*x2, y2, temp);
z = temp;
}
private:
T m_a, m_b, m_g;
};

template<class T>
struct PointwiseDivide
{
PointwiseDivide( T a, T b): m_a(a), m_b(b){}
DG_DEVICE
void operator()( T y, T& z)const{
T temp = z*m_b;
z = DG_FMA( m_a, z/y, temp);
}
DG_DEVICE
void operator()( T x, T y, T& z)const{
T temp = z*m_b;
z = DG_FMA( m_a, x/y, temp);
}
private:
T m_a, m_b;
};

namespace detail
{
template<class F, class G>
struct Compose
{
Compose( F f, G g):m_f(f), m_g(g){}
template<class ...Xs>
auto operator() ( Xs&& ... xs){
return m_f(m_g(std::forward<Xs>(xs)...));
}
template<class ...Xs>
auto operator() ( Xs&& ... xs) const {
return m_f(m_g(std::forward<Xs>(xs)...));
}
private:
F m_f;
G m_g;
};
}


template <class UnaryOp, class Functor>
auto compose( UnaryOp f, Functor g) {
return detail::Compose<UnaryOp,Functor>( f, g);
}

template <class UnaryOp, typename... Functors>
auto compose(UnaryOp f0, Functors... fs) {
return compose( f0 , compose(fs...));
}

}
