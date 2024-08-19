#pragma once
namespace glm{
namespace detail
{
template <typename T, int N>
struct _swizzle_base0
{
typedef T       value_type;
protected:
GLM_FUNC_QUALIFIER value_type&         elem   (size_t i)       { return (reinterpret_cast<value_type*>(_buffer))[i]; }
GLM_FUNC_QUALIFIER const value_type&   elem   (size_t i) const { return (reinterpret_cast<const value_type*>(_buffer))[i]; }
char    _buffer[1];
};
template <typename T, precision P, typename V, int E0, int E1, int E2, int E3, int N>
struct _swizzle_base1 : public _swizzle_base0<T, N>
{
};
template <typename T, precision P, typename V, int E0, int E1>
struct _swizzle_base1<T, P, V,E0,E1,-1,-2,2> : public _swizzle_base0<T, 2>
{
GLM_FUNC_QUALIFIER V operator ()()  const { return V(this->elem(E0), this->elem(E1)); }
};
template <typename T, precision P, typename V, int E0, int E1, int E2>
struct _swizzle_base1<T, P, V,E0,E1,E2,-1,3> : public _swizzle_base0<T, 3>
{
GLM_FUNC_QUALIFIER V operator ()()  const { return V(this->elem(E0), this->elem(E1), this->elem(E2)); }
};
template <typename T, precision P, typename V, int E0, int E1, int E2, int E3>
struct _swizzle_base1<T, P, V,E0,E1,E2,E3,4> : public _swizzle_base0<T, 4>
{ 
GLM_FUNC_QUALIFIER V operator ()()  const { return V(this->elem(E0), this->elem(E1), this->elem(E2), this->elem(E3)); }
};
template <typename ValueType, precision P, typename VecType, int N, int E0, int E1, int E2, int E3, int DUPLICATE_ELEMENTS>
struct _swizzle_base2 : public _swizzle_base1<ValueType, P, VecType,E0,E1,E2,E3,N>
{
typedef VecType vec_type;
typedef ValueType value_type;
GLM_FUNC_QUALIFIER _swizzle_base2& operator= (const ValueType& t)
{
for (int i = 0; i < N; ++i)
(*this)[i] = t;
return *this;
}
GLM_FUNC_QUALIFIER _swizzle_base2& operator= (const VecType& that)
{
struct op { 
GLM_FUNC_QUALIFIER void operator() (value_type& e, value_type& t) { e = t; } 
};
_apply_op(that, op());
return *this;
}
GLM_FUNC_QUALIFIER void operator -= (const VecType& that)
{
struct op { 
GLM_FUNC_QUALIFIER void operator() (value_type& e, value_type& t) { e -= t; } 
};
_apply_op(that, op());
}
GLM_FUNC_QUALIFIER void operator += (const VecType& that)
{
struct op { 
GLM_FUNC_QUALIFIER void operator() (value_type& e, value_type& t) { e += t; } 
};
_apply_op(that, op());
}
GLM_FUNC_QUALIFIER void operator *= (const VecType& that)
{
struct op { 
GLM_FUNC_QUALIFIER void operator() (value_type& e, value_type& t) { e *= t; } 
};
_apply_op(that, op());
}
GLM_FUNC_QUALIFIER void operator /= (const VecType& that)
{
struct op { 
GLM_FUNC_QUALIFIER void operator() (value_type& e, value_type& t) { e /= t; } 
};
_apply_op(that, op());
}
GLM_FUNC_QUALIFIER value_type& operator[]  (size_t i)
{
const int offset_dst[4] = { E0, E1, E2, E3 };
return this->elem(offset_dst[i]);
}
GLM_FUNC_QUALIFIER value_type  operator[]  (size_t i) const
{
const int offset_dst[4] = { E0, E1, E2, E3 };
return this->elem(offset_dst[i]);
}
protected:
template <typename T>
GLM_FUNC_QUALIFIER void _apply_op(const VecType& that, T op)
{
ValueType t[N];
for (int i = 0; i < N; ++i)
t[i] = that[i];
for (int i = 0; i < N; ++i)
op( (*this)[i], t[i] );
}
};
template <typename ValueType, precision P, typename VecType, int N, int E0, int E1, int E2, int E3>
struct _swizzle_base2<ValueType, P, VecType,N,E0,E1,E2,E3,1> : public _swizzle_base1<ValueType, P, VecType,E0,E1,E2,E3,N>
{
typedef VecType         vec_type;        
typedef ValueType       value_type;
struct Stub {};
GLM_FUNC_QUALIFIER _swizzle_base2& operator= (Stub const &) { return *this; }
GLM_FUNC_QUALIFIER value_type  operator[]  (size_t i) const
{
const int offset_dst[4] = { E0, E1, E2, E3 };
return this->elem(offset_dst[i]);
}
};
template <int N,typename ValueType, precision P, typename VecType, int E0,int E1,int E2,int E3>
struct _swizzle : public _swizzle_base2<ValueType, P, VecType, N, E0, E1, E2, E3, (E0==E1||E0==E2||E0==E3||E1==E2||E1==E3||E2==E3)>
{
typedef _swizzle_base2<ValueType, P, VecType,N,E0,E1,E2,E3,(E0==E1||E0==E2||E0==E3||E1==E2||E1==E3||E2==E3)> base_type;
using base_type::operator=;
GLM_FUNC_QUALIFIER operator VecType () const { return (*this)(); }
};
#define _GLM_SWIZZLE_TEMPLATE1   template <int N, typename T, precision P, typename V, int E0, int E1, int E2, int E3>
#define _GLM_SWIZZLE_TEMPLATE2   template <int N, typename T, precision P, typename V, int E0, int E1, int E2, int E3, int F0, int F1, int F2, int F3>
#define _GLM_SWIZZLE_TYPE1       _swizzle<N, T, P, V, E0, E1, E2, E3>
#define _GLM_SWIZZLE_TYPE2       _swizzle<N, T, P, V, F0, F1, F2, F3>
#define _GLM_SWIZZLE_VECTOR_BINARY_OPERATOR_IMPLEMENTATION(OPERAND)                 \
_GLM_SWIZZLE_TEMPLATE2                                                          \
GLM_FUNC_QUALIFIER V operator OPERAND ( const _GLM_SWIZZLE_TYPE1& a, const _GLM_SWIZZLE_TYPE2& b)  \
{                                                                               \
return a() OPERAND b();                                                     \
}                                                                               \
_GLM_SWIZZLE_TEMPLATE1                                                          \
GLM_FUNC_QUALIFIER V operator OPERAND ( const _GLM_SWIZZLE_TYPE1& a, const V& b)                   \
{                                                                               \
return a() OPERAND b;                                                       \
}                                                                               \
_GLM_SWIZZLE_TEMPLATE1                                                          \
GLM_FUNC_QUALIFIER V operator OPERAND ( const V& a, const _GLM_SWIZZLE_TYPE1& b)                   \
{                                                                               \
return a OPERAND b();                                                       \
}
#define _GLM_SWIZZLE_SCALAR_BINARY_OPERATOR_IMPLEMENTATION(OPERAND)                 \
_GLM_SWIZZLE_TEMPLATE1                                                          \
GLM_FUNC_QUALIFIER V operator OPERAND ( const _GLM_SWIZZLE_TYPE1& a, const T& b)                   \
{                                                                               \
return a() OPERAND b;                                                       \
}                                                                               \
_GLM_SWIZZLE_TEMPLATE1                                                          \
GLM_FUNC_QUALIFIER V operator OPERAND ( const T& a, const _GLM_SWIZZLE_TYPE1& b)                   \
{                                                                               \
return a OPERAND b();                                                       \
}
#define _GLM_SWIZZLE_FUNCTION_1_ARGS(RETURN_TYPE,FUNCTION)                          \
_GLM_SWIZZLE_TEMPLATE1                                                          \
GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const _GLM_SWIZZLE_TYPE1& a)  \
{                                                                               \
return FUNCTION(a());                                                       \
}
#define _GLM_SWIZZLE_FUNCTION_2_ARGS(RETURN_TYPE,FUNCTION)                                                      \
_GLM_SWIZZLE_TEMPLATE2                                                                                      \
GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const _GLM_SWIZZLE_TYPE1& a, const _GLM_SWIZZLE_TYPE2& b) \
{                                                                                                           \
return FUNCTION(a(), b());                                                                              \
}                                                                                                           \
_GLM_SWIZZLE_TEMPLATE1                                                                                      \
GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const _GLM_SWIZZLE_TYPE1& a, const _GLM_SWIZZLE_TYPE1& b) \
{                                                                                                           \
return FUNCTION(a(), b());                                                                              \
}                                                                                                           \
_GLM_SWIZZLE_TEMPLATE1                                                                                      \
GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const _GLM_SWIZZLE_TYPE1& a, const typename V& b)         \
{                                                                                                           \
return FUNCTION(a(), b);                                                                                \
}                                                                                                           \
_GLM_SWIZZLE_TEMPLATE1                                                                                      \
GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const V& a, const _GLM_SWIZZLE_TYPE1& b)                  \
{                                                                                                           \
return FUNCTION(a, b());                                                                                \
} 
#define _GLM_SWIZZLE_FUNCTION_2_ARGS_SCALAR(RETURN_TYPE,FUNCTION)                                                             \
_GLM_SWIZZLE_TEMPLATE2                                                                                                    \
GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const _GLM_SWIZZLE_TYPE1& a, const _GLM_SWIZZLE_TYPE2& b, const T& c)   \
{                                                                                                                         \
return FUNCTION(a(), b(), c);                                                                                         \
}                                                                                                                         \
_GLM_SWIZZLE_TEMPLATE1                                                                                                    \
GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const _GLM_SWIZZLE_TYPE1& a, const _GLM_SWIZZLE_TYPE1& b, const T& c)   \
{                                                                                                                         \
return FUNCTION(a(), b(), c);                                                                                         \
}                                                                                                                         \
_GLM_SWIZZLE_TEMPLATE1                                                                                                    \
GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const _GLM_SWIZZLE_TYPE1& a, const typename S0::vec_type& b, const T& c)\
{                                                                                                                         \
return FUNCTION(a(), b, c);                                                                                           \
}                                                                                                                         \
_GLM_SWIZZLE_TEMPLATE1                                                                                                    \
GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const typename V& a, const _GLM_SWIZZLE_TYPE1& b, const T& c)           \
{                                                                                                                         \
return FUNCTION(a, b(), c);                                                                                           \
} 
}
}
namespace glm
{
namespace detail
{
_GLM_SWIZZLE_SCALAR_BINARY_OPERATOR_IMPLEMENTATION(-)
_GLM_SWIZZLE_SCALAR_BINARY_OPERATOR_IMPLEMENTATION(*)
_GLM_SWIZZLE_VECTOR_BINARY_OPERATOR_IMPLEMENTATION(+)
_GLM_SWIZZLE_VECTOR_BINARY_OPERATOR_IMPLEMENTATION(-)
_GLM_SWIZZLE_VECTOR_BINARY_OPERATOR_IMPLEMENTATION(*)
_GLM_SWIZZLE_VECTOR_BINARY_OPERATOR_IMPLEMENTATION(/)
}
}
#define _GLM_SWIZZLE2_2_MEMBERS(T, P, V, E0,E1) \
struct { detail::_swizzle<2, T, P, V<T, P>, 0,0,-1,-2> E0 ## E0; }; \
struct { detail::_swizzle<2, T, P, V<T, P>, 0,1,-1,-2> E0 ## E1; }; \
struct { detail::_swizzle<2, T, P, V<T, P>, 1,0,-1,-2> E1 ## E0; }; \
struct { detail::_swizzle<2, T, P, V<T, P>, 1,1,-1,-2> E1 ## E1; }; 
#define _GLM_SWIZZLE2_3_MEMBERS(T, P, V, E0,E1) \
struct { detail::_swizzle<3,T, P, V<T, P>, 0,0,0,-1> E0 ## E0 ## E0; }; \
struct { detail::_swizzle<3,T, P, V<T, P>, 0,0,1,-1> E0 ## E0 ## E1; }; \
struct { detail::_swizzle<3,T, P, V<T, P>, 0,1,0,-1> E0 ## E1 ## E0; }; \
struct { detail::_swizzle<3,T, P, V<T, P>, 0,1,1,-1> E0 ## E1 ## E1; }; \
struct { detail::_swizzle<3,T, P, V<T, P>, 1,0,0,-1> E1 ## E0 ## E0; }; \
struct { detail::_swizzle<3,T, P, V<T, P>, 1,0,1,-1> E1 ## E0 ## E1; }; \
struct { detail::_swizzle<3,T, P, V<T, P>, 1,1,0,-1> E1 ## E1 ## E0; }; \
struct { detail::_swizzle<3,T, P, V<T, P>, 1,1,1,-1> E1 ## E1 ## E1; };  
#define _GLM_SWIZZLE2_4_MEMBERS(T, P, V, E0,E1) \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,0,0> E0 ## E0 ## E0 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,0,1> E0 ## E0 ## E0 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,1,0> E0 ## E0 ## E1 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,1,1> E0 ## E0 ## E1 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,0,0> E0 ## E1 ## E0 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,0,1> E0 ## E1 ## E0 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,1,0> E0 ## E1 ## E1 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,1,1> E0 ## E1 ## E1 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,0,0> E1 ## E0 ## E0 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,0,1> E1 ## E0 ## E0 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,1,0> E1 ## E0 ## E1 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,1,1> E1 ## E0 ## E1 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,0,0> E1 ## E1 ## E0 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,0,1> E1 ## E1 ## E0 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,1,0> E1 ## E1 ## E1 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,1,1> E1 ## E1 ## E1 ## E1; };  
#define _GLM_SWIZZLE3_2_MEMBERS(T, P, V, E0,E1,E2) \
struct { detail::_swizzle<2,T, P, V<T, P>, 0,0,-1,-2> E0 ## E0; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 0,1,-1,-2> E0 ## E1; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 0,2,-1,-2> E0 ## E2; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 1,0,-1,-2> E1 ## E0; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 1,1,-1,-2> E1 ## E1; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 1,2,-1,-2> E1 ## E2; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 2,0,-1,-2> E2 ## E0; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 2,1,-1,-2> E2 ## E1; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 2,2,-1,-2> E2 ## E2; }; 
#define _GLM_SWIZZLE3_3_MEMBERS(T, P, V ,E0,E1,E2) \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,0,0,-1> E0 ## E0 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,0,1,-1> E0 ## E0 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,0,2,-1> E0 ## E0 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,1,0,-1> E0 ## E1 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,1,1,-1> E0 ## E1 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,1,2,-1> E0 ## E1 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,2,0,-1> E0 ## E2 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,2,1,-1> E0 ## E2 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,2,2,-1> E0 ## E2 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,0,0,-1> E1 ## E0 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,0,1,-1> E1 ## E0 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,0,2,-1> E1 ## E0 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,1,0,-1> E1 ## E1 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,1,1,-1> E1 ## E1 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,1,2,-1> E1 ## E1 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,2,0,-1> E1 ## E2 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,2,1,-1> E1 ## E2 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,2,2,-1> E1 ## E2 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,0,0,-1> E2 ## E0 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,0,1,-1> E2 ## E0 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,0,2,-1> E2 ## E0 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,1,0,-1> E2 ## E1 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,1,1,-1> E2 ## E1 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,1,2,-1> E2 ## E1 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,2,0,-1> E2 ## E2 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,2,1,-1> E2 ## E2 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,2,2,-1> E2 ## E2 ## E2; };
#define _GLM_SWIZZLE3_4_MEMBERS(T, P, V, E0,E1,E2) \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,0,0> E0 ## E0 ## E0 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,0,1> E0 ## E0 ## E0 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,0,2> E0 ## E0 ## E0 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,1,0> E0 ## E0 ## E1 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,1,1> E0 ## E0 ## E1 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,1,2> E0 ## E0 ## E1 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,2,0> E0 ## E0 ## E2 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,2,1> E0 ## E0 ## E2 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,2,2> E0 ## E0 ## E2 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,0,0> E0 ## E1 ## E0 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,0,1> E0 ## E1 ## E0 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,0,2> E0 ## E1 ## E0 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,1,0> E0 ## E1 ## E1 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,1,1> E0 ## E1 ## E1 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,1,2> E0 ## E1 ## E1 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,2,0> E0 ## E1 ## E2 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,2,1> E0 ## E1 ## E2 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,2,2> E0 ## E1 ## E2 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,0,0> E0 ## E2 ## E0 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,0,1> E0 ## E2 ## E0 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,0,2> E0 ## E2 ## E0 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,1,0> E0 ## E2 ## E1 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,1,1> E0 ## E2 ## E1 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,1,2> E0 ## E2 ## E1 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,2,0> E0 ## E2 ## E2 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,2,1> E0 ## E2 ## E2 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,2,2> E0 ## E2 ## E2 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,0,0> E1 ## E0 ## E0 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,0,1> E1 ## E0 ## E0 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,0,2> E1 ## E0 ## E0 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,1,0> E1 ## E0 ## E1 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,1,1> E1 ## E0 ## E1 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,1,2> E1 ## E0 ## E1 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,2,0> E1 ## E0 ## E2 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,2,1> E1 ## E0 ## E2 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,2,2> E1 ## E0 ## E2 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,0,0> E1 ## E1 ## E0 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,0,1> E1 ## E1 ## E0 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,0,2> E1 ## E1 ## E0 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,1,0> E1 ## E1 ## E1 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,1,1> E1 ## E1 ## E1 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,1,2> E1 ## E1 ## E1 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,2,0> E1 ## E1 ## E2 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,2,1> E1 ## E1 ## E2 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,2,2> E1 ## E1 ## E2 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,0,0> E1 ## E2 ## E0 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,0,1> E1 ## E2 ## E0 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,0,2> E1 ## E2 ## E0 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,1,0> E1 ## E2 ## E1 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,1,1> E1 ## E2 ## E1 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,1,2> E1 ## E2 ## E1 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,2,0> E1 ## E2 ## E2 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,2,1> E1 ## E2 ## E2 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,2,2> E1 ## E2 ## E2 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,0,0> E2 ## E0 ## E0 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,0,1> E2 ## E0 ## E0 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,0,2> E2 ## E0 ## E0 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,1,0> E2 ## E0 ## E1 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,1,1> E2 ## E0 ## E1 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,1,2> E2 ## E0 ## E1 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,2,0> E2 ## E0 ## E2 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,2,1> E2 ## E0 ## E2 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,2,2> E2 ## E0 ## E2 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,0,0> E2 ## E1 ## E0 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,0,1> E2 ## E1 ## E0 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,0,2> E2 ## E1 ## E0 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,1,0> E2 ## E1 ## E1 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,1,1> E2 ## E1 ## E1 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,1,2> E2 ## E1 ## E1 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,2,0> E2 ## E1 ## E2 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,2,1> E2 ## E1 ## E2 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,2,2> E2 ## E1 ## E2 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,0,0> E2 ## E2 ## E0 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,0,1> E2 ## E2 ## E0 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,0,2> E2 ## E2 ## E0 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,1,0> E2 ## E2 ## E1 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,1,1> E2 ## E2 ## E1 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,1,2> E2 ## E2 ## E1 ## E2; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,2,0> E2 ## E2 ## E2 ## E0; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,2,1> E2 ## E2 ## E2 ## E1; }; \
struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,2,2> E2 ## E2 ## E2 ## E2; }; 
#define _GLM_SWIZZLE4_2_MEMBERS(T, P, V, E0,E1,E2,E3) \
struct { detail::_swizzle<2,T, P, V<T, P>, 0,0,-1,-2> E0 ## E0; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 0,1,-1,-2> E0 ## E1; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 0,2,-1,-2> E0 ## E2; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 0,3,-1,-2> E0 ## E3; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 1,0,-1,-2> E1 ## E0; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 1,1,-1,-2> E1 ## E1; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 1,2,-1,-2> E1 ## E2; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 1,3,-1,-2> E1 ## E3; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 2,0,-1,-2> E2 ## E0; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 2,1,-1,-2> E2 ## E1; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 2,2,-1,-2> E2 ## E2; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 2,3,-1,-2> E2 ## E3; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 3,0,-1,-2> E3 ## E0; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 3,1,-1,-2> E3 ## E1; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 3,2,-1,-2> E3 ## E2; }; \
struct { detail::_swizzle<2,T, P, V<T, P>, 3,3,-1,-2> E3 ## E3; }; 
#define _GLM_SWIZZLE4_3_MEMBERS(T,P, V, E0,E1,E2,E3) \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,0,0,-1> E0 ## E0 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,0,1,-1> E0 ## E0 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,0,2,-1> E0 ## E0 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,0,3,-1> E0 ## E0 ## E3; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,1,0,-1> E0 ## E1 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,1,1,-1> E0 ## E1 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,1,2,-1> E0 ## E1 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,1,3,-1> E0 ## E1 ## E3; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,2,0,-1> E0 ## E2 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,2,1,-1> E0 ## E2 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,2,2,-1> E0 ## E2 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,2,3,-1> E0 ## E2 ## E3; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,3,0,-1> E0 ## E3 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,3,1,-1> E0 ## E3 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,3,2,-1> E0 ## E3 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 0,3,3,-1> E0 ## E3 ## E3; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,0,0,-1> E1 ## E0 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,0,1,-1> E1 ## E0 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,0,2,-1> E1 ## E0 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,0,3,-1> E1 ## E0 ## E3; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,1,0,-1> E1 ## E1 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,1,1,-1> E1 ## E1 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,1,2,-1> E1 ## E1 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,1,3,-1> E1 ## E1 ## E3; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,2,0,-1> E1 ## E2 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,2,1,-1> E1 ## E2 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,2,2,-1> E1 ## E2 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,2,3,-1> E1 ## E2 ## E3; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,3,0,-1> E1 ## E3 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,3,1,-1> E1 ## E3 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,3,2,-1> E1 ## E3 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 1,3,3,-1> E1 ## E3 ## E3; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,0,0,-1> E2 ## E0 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,0,1,-1> E2 ## E0 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,0,2,-1> E2 ## E0 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,0,3,-1> E2 ## E0 ## E3; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,1,0,-1> E2 ## E1 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,1,1,-1> E2 ## E1 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,1,2,-1> E2 ## E1 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,1,3,-1> E2 ## E1 ## E3; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,2,0,-1> E2 ## E2 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,2,1,-1> E2 ## E2 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,2,2,-1> E2 ## E2 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,2,3,-1> E2 ## E2 ## E3; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,3,0,-1> E2 ## E3 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,3,1,-1> E2 ## E3 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,3,2,-1> E2 ## E3 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 2,3,3,-1> E2 ## E3 ## E3; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 3,0,0,-1> E3 ## E0 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 3,0,1,-1> E3 ## E0 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 3,0,2,-1> E3 ## E0 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 3,0,3,-1> E3 ## E0 ## E3; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 3,1,0,-1> E3 ## E1 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 3,1,1,-1> E3 ## E1 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 3,1,2,-1> E3 ## E1 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 3,1,3,-1> E3 ## E1 ## E3; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 3,2,0,-1> E3 ## E2 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 3,2,1,-1> E3 ## E2 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 3,2,2,-1> E3 ## E2 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 3,2,3,-1> E3 ## E2 ## E3; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 3,3,0,-1> E3 ## E3 ## E0; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 3,3,1,-1> E3 ## E3 ## E1; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 3,3,2,-1> E3 ## E3 ## E2; }; \
struct { detail::_swizzle<3,T,P, V<T, P>, 3,3,3,-1> E3 ## E3 ## E3; };  
#define _GLM_SWIZZLE4_4_MEMBERS(T, P, V, E0,E1,E2,E3) \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,0,0> E0 ## E0 ## E0 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,0,1> E0 ## E0 ## E0 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,0,2> E0 ## E0 ## E0 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,0,3> E0 ## E0 ## E0 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,1,0> E0 ## E0 ## E1 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,1,1> E0 ## E0 ## E1 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,1,2> E0 ## E0 ## E1 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,1,3> E0 ## E0 ## E1 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,2,0> E0 ## E0 ## E2 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,2,1> E0 ## E0 ## E2 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,2,2> E0 ## E0 ## E2 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,2,3> E0 ## E0 ## E2 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,3,0> E0 ## E0 ## E3 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,3,1> E0 ## E0 ## E3 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,3,2> E0 ## E0 ## E3 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,3,3> E0 ## E0 ## E3 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,0,0> E0 ## E1 ## E0 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,0,1> E0 ## E1 ## E0 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,0,2> E0 ## E1 ## E0 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,0,3> E0 ## E1 ## E0 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,1,0> E0 ## E1 ## E1 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,1,1> E0 ## E1 ## E1 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,1,2> E0 ## E1 ## E1 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,1,3> E0 ## E1 ## E1 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,2,0> E0 ## E1 ## E2 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,2,1> E0 ## E1 ## E2 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,2,2> E0 ## E1 ## E2 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,2,3> E0 ## E1 ## E2 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,3,0> E0 ## E1 ## E3 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,3,1> E0 ## E1 ## E3 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,3,2> E0 ## E1 ## E3 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,3,3> E0 ## E1 ## E3 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,0,0> E0 ## E2 ## E0 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,0,1> E0 ## E2 ## E0 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,0,2> E0 ## E2 ## E0 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,0,3> E0 ## E2 ## E0 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,1,0> E0 ## E2 ## E1 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,1,1> E0 ## E2 ## E1 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,1,2> E0 ## E2 ## E1 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,1,3> E0 ## E2 ## E1 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,2,0> E0 ## E2 ## E2 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,2,1> E0 ## E2 ## E2 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,2,2> E0 ## E2 ## E2 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,2,3> E0 ## E2 ## E2 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,3,0> E0 ## E2 ## E3 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,3,1> E0 ## E2 ## E3 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,3,2> E0 ## E2 ## E3 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,3,3> E0 ## E2 ## E3 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,0,0> E0 ## E3 ## E0 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,0,1> E0 ## E3 ## E0 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,0,2> E0 ## E3 ## E0 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,0,3> E0 ## E3 ## E0 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,1,0> E0 ## E3 ## E1 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,1,1> E0 ## E3 ## E1 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,1,2> E0 ## E3 ## E1 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,1,3> E0 ## E3 ## E1 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,2,0> E0 ## E3 ## E2 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,2,1> E0 ## E3 ## E2 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,2,2> E0 ## E3 ## E2 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,2,3> E0 ## E3 ## E2 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,3,0> E0 ## E3 ## E3 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,3,1> E0 ## E3 ## E3 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,3,2> E0 ## E3 ## E3 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,3,3> E0 ## E3 ## E3 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,0,0> E1 ## E0 ## E0 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,0,1> E1 ## E0 ## E0 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,0,2> E1 ## E0 ## E0 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,0,3> E1 ## E0 ## E0 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,1,0> E1 ## E0 ## E1 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,1,1> E1 ## E0 ## E1 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,1,2> E1 ## E0 ## E1 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,1,3> E1 ## E0 ## E1 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,2,0> E1 ## E0 ## E2 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,2,1> E1 ## E0 ## E2 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,2,2> E1 ## E0 ## E2 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,2,3> E1 ## E0 ## E2 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,3,0> E1 ## E0 ## E3 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,3,1> E1 ## E0 ## E3 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,3,2> E1 ## E0 ## E3 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,3,3> E1 ## E0 ## E3 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,0,0> E1 ## E1 ## E0 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,0,1> E1 ## E1 ## E0 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,0,2> E1 ## E1 ## E0 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,0,3> E1 ## E1 ## E0 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,1,0> E1 ## E1 ## E1 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,1,1> E1 ## E1 ## E1 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,1,2> E1 ## E1 ## E1 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,1,3> E1 ## E1 ## E1 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,2,0> E1 ## E1 ## E2 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,2,1> E1 ## E1 ## E2 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,2,2> E1 ## E1 ## E2 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,2,3> E1 ## E1 ## E2 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,3,0> E1 ## E1 ## E3 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,3,1> E1 ## E1 ## E3 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,3,2> E1 ## E1 ## E3 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,3,3> E1 ## E1 ## E3 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,0,0> E1 ## E2 ## E0 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,0,1> E1 ## E2 ## E0 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,0,2> E1 ## E2 ## E0 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,0,3> E1 ## E2 ## E0 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,1,0> E1 ## E2 ## E1 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,1,1> E1 ## E2 ## E1 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,1,2> E1 ## E2 ## E1 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,1,3> E1 ## E2 ## E1 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,2,0> E1 ## E2 ## E2 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,2,1> E1 ## E2 ## E2 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,2,2> E1 ## E2 ## E2 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,2,3> E1 ## E2 ## E2 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,3,0> E1 ## E2 ## E3 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,3,1> E1 ## E2 ## E3 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,3,2> E1 ## E2 ## E3 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,3,3> E1 ## E2 ## E3 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,0,0> E1 ## E3 ## E0 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,0,1> E1 ## E3 ## E0 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,0,2> E1 ## E3 ## E0 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,0,3> E1 ## E3 ## E0 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,1,0> E1 ## E3 ## E1 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,1,1> E1 ## E3 ## E1 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,1,2> E1 ## E3 ## E1 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,1,3> E1 ## E3 ## E1 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,2,0> E1 ## E3 ## E2 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,2,1> E1 ## E3 ## E2 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,2,2> E1 ## E3 ## E2 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,2,3> E1 ## E3 ## E2 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,3,0> E1 ## E3 ## E3 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,3,1> E1 ## E3 ## E3 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,3,2> E1 ## E3 ## E3 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,3,3> E1 ## E3 ## E3 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,0,0> E2 ## E0 ## E0 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,0,1> E2 ## E0 ## E0 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,0,2> E2 ## E0 ## E0 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,0,3> E2 ## E0 ## E0 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,1,0> E2 ## E0 ## E1 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,1,1> E2 ## E0 ## E1 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,1,2> E2 ## E0 ## E1 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,1,3> E2 ## E0 ## E1 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,2,0> E2 ## E0 ## E2 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,2,1> E2 ## E0 ## E2 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,2,2> E2 ## E0 ## E2 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,2,3> E2 ## E0 ## E2 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,3,0> E2 ## E0 ## E3 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,3,1> E2 ## E0 ## E3 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,3,2> E2 ## E0 ## E3 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,3,3> E2 ## E0 ## E3 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,0,0> E2 ## E1 ## E0 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,0,1> E2 ## E1 ## E0 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,0,2> E2 ## E1 ## E0 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,0,3> E2 ## E1 ## E0 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,1,0> E2 ## E1 ## E1 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,1,1> E2 ## E1 ## E1 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,1,2> E2 ## E1 ## E1 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,1,3> E2 ## E1 ## E1 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,2,0> E2 ## E1 ## E2 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,2,1> E2 ## E1 ## E2 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,2,2> E2 ## E1 ## E2 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,2,3> E2 ## E1 ## E2 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,3,0> E2 ## E1 ## E3 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,3,1> E2 ## E1 ## E3 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,3,2> E2 ## E1 ## E3 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,3,3> E2 ## E1 ## E3 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,0,0> E2 ## E2 ## E0 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,0,1> E2 ## E2 ## E0 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,0,2> E2 ## E2 ## E0 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,0,3> E2 ## E2 ## E0 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,1,0> E2 ## E2 ## E1 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,1,1> E2 ## E2 ## E1 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,1,2> E2 ## E2 ## E1 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,1,3> E2 ## E2 ## E1 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,2,0> E2 ## E2 ## E2 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,2,1> E2 ## E2 ## E2 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,2,2> E2 ## E2 ## E2 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,2,3> E2 ## E2 ## E2 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,3,0> E2 ## E2 ## E3 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,3,1> E2 ## E2 ## E3 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,3,2> E2 ## E2 ## E3 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,3,3> E2 ## E2 ## E3 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,0,0> E2 ## E3 ## E0 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,0,1> E2 ## E3 ## E0 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,0,2> E2 ## E3 ## E0 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,0,3> E2 ## E3 ## E0 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,1,0> E2 ## E3 ## E1 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,1,1> E2 ## E3 ## E1 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,1,2> E2 ## E3 ## E1 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,1,3> E2 ## E3 ## E1 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,2,0> E2 ## E3 ## E2 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,2,1> E2 ## E3 ## E2 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,2,2> E2 ## E3 ## E2 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,2,3> E2 ## E3 ## E2 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,3,0> E2 ## E3 ## E3 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,3,1> E2 ## E3 ## E3 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,3,2> E2 ## E3 ## E3 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,3,3> E2 ## E3 ## E3 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,0,0> E3 ## E0 ## E0 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,0,1> E3 ## E0 ## E0 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,0,2> E3 ## E0 ## E0 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,0,3> E3 ## E0 ## E0 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,1,0> E3 ## E0 ## E1 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,1,1> E3 ## E0 ## E1 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,1,2> E3 ## E0 ## E1 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,1,3> E3 ## E0 ## E1 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,2,0> E3 ## E0 ## E2 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,2,1> E3 ## E0 ## E2 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,2,2> E3 ## E0 ## E2 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,2,3> E3 ## E0 ## E2 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,3,0> E3 ## E0 ## E3 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,3,1> E3 ## E0 ## E3 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,3,2> E3 ## E0 ## E3 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,3,3> E3 ## E0 ## E3 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,0,0> E3 ## E1 ## E0 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,0,1> E3 ## E1 ## E0 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,0,2> E3 ## E1 ## E0 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,0,3> E3 ## E1 ## E0 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,1,0> E3 ## E1 ## E1 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,1,1> E3 ## E1 ## E1 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,1,2> E3 ## E1 ## E1 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,1,3> E3 ## E1 ## E1 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,2,0> E3 ## E1 ## E2 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,2,1> E3 ## E1 ## E2 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,2,2> E3 ## E1 ## E2 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,2,3> E3 ## E1 ## E2 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,3,0> E3 ## E1 ## E3 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,3,1> E3 ## E1 ## E3 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,3,2> E3 ## E1 ## E3 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,3,3> E3 ## E1 ## E3 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,0,0> E3 ## E2 ## E0 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,0,1> E3 ## E2 ## E0 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,0,2> E3 ## E2 ## E0 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,0,3> E3 ## E2 ## E0 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,1,0> E3 ## E2 ## E1 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,1,1> E3 ## E2 ## E1 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,1,2> E3 ## E2 ## E1 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,1,3> E3 ## E2 ## E1 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,2,0> E3 ## E2 ## E2 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,2,1> E3 ## E2 ## E2 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,2,2> E3 ## E2 ## E2 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,2,3> E3 ## E2 ## E2 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,3,0> E3 ## E2 ## E3 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,3,1> E3 ## E2 ## E3 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,3,2> E3 ## E2 ## E3 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,3,3> E3 ## E2 ## E3 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,0,0> E3 ## E3 ## E0 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,0,1> E3 ## E3 ## E0 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,0,2> E3 ## E3 ## E0 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,0,3> E3 ## E3 ## E0 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,1,0> E3 ## E3 ## E1 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,1,1> E3 ## E3 ## E1 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,1,2> E3 ## E3 ## E1 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,1,3> E3 ## E3 ## E1 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,2,0> E3 ## E3 ## E2 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,2,1> E3 ## E3 ## E2 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,2,2> E3 ## E3 ## E2 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,2,3> E3 ## E3 ## E2 ## E3; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,3,0> E3 ## E3 ## E3 ## E0; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,3,1> E3 ## E3 ## E3 ## E1; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,3,2> E3 ## E3 ## E3 ## E2; }; \
struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,3,3> E3 ## E3 ## E3 ## E3; };
