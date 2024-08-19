




#ifndef BOOST_OPERATORS_V1_HPP
#define BOOST_OPERATORS_V1_HPP

#include <cstddef>
#include <iterator>

#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>

#if defined(__sgi) && !defined(__GNUC__)
#   pragma set woff 1234
#endif

#if BOOST_WORKAROUND(BOOST_MSVC, < 1600)
#   pragma warning( disable : 4284 ) 
#endif                               

namespace boost {
namespace detail {

template <typename T> class empty_base {};

} 
} 


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace boost
{
#endif



template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct less_than_comparable2 : B
{
friend bool operator<=(const T& x, const U& y) { return !static_cast<bool>(x > y); }
friend bool operator>=(const T& x, const U& y) { return !static_cast<bool>(x < y); }
friend bool operator>(const U& x, const T& y)  { return y < x; }
friend bool operator<(const U& x, const T& y)  { return y > x; }
friend bool operator<=(const U& x, const T& y) { return !static_cast<bool>(y < x); }
friend bool operator>=(const U& x, const T& y) { return !static_cast<bool>(y > x); }
};

template <class T, class B = ::boost::detail::empty_base<T> >
struct less_than_comparable1 : B
{
friend bool operator>(const T& x, const T& y)  { return y < x; }
friend bool operator<=(const T& x, const T& y) { return !static_cast<bool>(y < x); }
friend bool operator>=(const T& x, const T& y) { return !static_cast<bool>(x < y); }
};

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct equality_comparable2 : B
{
friend bool operator==(const U& y, const T& x) { return x == y; }
friend bool operator!=(const U& y, const T& x) { return !static_cast<bool>(x == y); }
friend bool operator!=(const T& y, const U& x) { return !static_cast<bool>(y == x); }
};

template <class T, class B = ::boost::detail::empty_base<T> >
struct equality_comparable1 : B
{
friend bool operator!=(const T& x, const T& y) { return !static_cast<bool>(x == y); }
};

#define BOOST_OPERATOR2_LEFT(name) name##2##_##left


#if defined(BOOST_HAS_NRVO) || defined(BOOST_FORCE_SYMMETRIC_OPERATORS)


#define BOOST_BINARY_OPERATOR_COMMUTATIVE( NAME, OP )                         \
template <class T, class U, class B = ::boost::detail::empty_base<T> >        \
struct NAME##2 : B                                                            \
{                                                                             \
friend T operator OP( const T& lhs, const U& rhs )                          \
{ T nrv( lhs ); nrv OP##= rhs; return nrv; }                              \
friend T operator OP( const U& lhs, const T& rhs )                          \
{ T nrv( rhs ); nrv OP##= lhs; return nrv; }                              \
};                                                                            \
\
template <class T, class B = ::boost::detail::empty_base<T> >                 \
struct NAME##1 : B                                                            \
{                                                                             \
friend T operator OP( const T& lhs, const T& rhs )                          \
{ T nrv( lhs ); nrv OP##= rhs; return nrv; }                              \
};

#define BOOST_BINARY_OPERATOR_NON_COMMUTATIVE( NAME, OP )               \
template <class T, class U, class B = ::boost::detail::empty_base<T> >  \
struct NAME##2 : B                                                      \
{                                                                       \
friend T operator OP( const T& lhs, const U& rhs )                    \
{ T nrv( lhs ); nrv OP##= rhs; return nrv; }                        \
};                                                                      \
\
template <class T, class U, class B = ::boost::detail::empty_base<T> >  \
struct BOOST_OPERATOR2_LEFT(NAME) : B                                   \
{                                                                       \
friend T operator OP( const U& lhs, const T& rhs )                    \
{ T nrv( lhs ); nrv OP##= rhs; return nrv; }                        \
};                                                                      \
\
template <class T, class B = ::boost::detail::empty_base<T> >           \
struct NAME##1 : B                                                      \
{                                                                       \
friend T operator OP( const T& lhs, const T& rhs )                    \
{ T nrv( lhs ); nrv OP##= rhs; return nrv; }                        \
};

#else 


#define BOOST_BINARY_OPERATOR_COMMUTATIVE( NAME, OP )                   \
template <class T, class U, class B = ::boost::detail::empty_base<T> >  \
struct NAME##2 : B                                                      \
{                                                                       \
friend T operator OP( T lhs, const U& rhs ) { return lhs OP##= rhs; } \
friend T operator OP( const U& lhs, T rhs ) { return rhs OP##= lhs; } \
};                                                                      \
\
template <class T, class B = ::boost::detail::empty_base<T> >           \
struct NAME##1 : B                                                      \
{                                                                       \
friend T operator OP( T lhs, const T& rhs ) { return lhs OP##= rhs; } \
};

#define BOOST_BINARY_OPERATOR_NON_COMMUTATIVE( NAME, OP )               \
template <class T, class U, class B = ::boost::detail::empty_base<T> >  \
struct NAME##2 : B                                                      \
{                                                                       \
friend T operator OP( T lhs, const U& rhs ) { return lhs OP##= rhs; } \
};                                                                      \
\
template <class T, class U, class B = ::boost::detail::empty_base<T> >  \
struct BOOST_OPERATOR2_LEFT(NAME) : B                                   \
{                                                                       \
friend T operator OP( const U& lhs, const T& rhs )                    \
{ return T( lhs ) OP##= rhs; }                                      \
};                                                                      \
\
template <class T, class B = ::boost::detail::empty_base<T> >           \
struct NAME##1 : B                                                      \
{                                                                       \
friend T operator OP( T lhs, const T& rhs ) { return lhs OP##= rhs; } \
};

#endif 

BOOST_BINARY_OPERATOR_COMMUTATIVE( multipliable, * )
BOOST_BINARY_OPERATOR_COMMUTATIVE( addable, + )
BOOST_BINARY_OPERATOR_NON_COMMUTATIVE( subtractable, - )
BOOST_BINARY_OPERATOR_NON_COMMUTATIVE( dividable, / )
BOOST_BINARY_OPERATOR_NON_COMMUTATIVE( modable, % )
BOOST_BINARY_OPERATOR_COMMUTATIVE( xorable, ^ )
BOOST_BINARY_OPERATOR_COMMUTATIVE( andable, & )
BOOST_BINARY_OPERATOR_COMMUTATIVE( orable, | )

#undef BOOST_BINARY_OPERATOR_COMMUTATIVE
#undef BOOST_BINARY_OPERATOR_NON_COMMUTATIVE
#undef BOOST_OPERATOR2_LEFT


template <class T, class B = ::boost::detail::empty_base<T> >
struct incrementable : B
{
friend T operator++(T& x, int)
{
incrementable_type nrv(x);
++x;
return nrv;
}
private: 
typedef T incrementable_type;
};

template <class T, class B = ::boost::detail::empty_base<T> >
struct decrementable : B
{
friend T operator--(T& x, int)
{
decrementable_type nrv(x);
--x;
return nrv;
}
private: 
typedef T decrementable_type;
};


template <class T, class P, class B = ::boost::detail::empty_base<T> >
struct dereferenceable : B
{
P operator->() const
{ 
return &*static_cast<const T&>(*this); 
}
};

template <class T, class I, class R, class B = ::boost::detail::empty_base<T> >
struct indexable : B
{
R operator[](I n) const
{
return *(static_cast<const T&>(*this) + n);
}
};


#if defined(BOOST_HAS_NRVO) || defined(BOOST_FORCE_SYMMETRIC_OPERATORS)

#define BOOST_BINARY_OPERATOR( NAME, OP )                                     \
template <class T, class U, class B = ::boost::detail::empty_base<T> >        \
struct NAME##2 : B                                                            \
{                                                                             \
friend T operator OP( const T& lhs, const U& rhs )                          \
{ T nrv( lhs ); nrv OP##= rhs; return nrv; }                              \
};                                                                            \
\
template <class T, class B = ::boost::detail::empty_base<T> >                 \
struct NAME##1 : B                                                            \
{                                                                             \
friend T operator OP( const T& lhs, const T& rhs )                          \
{ T nrv( lhs ); nrv OP##= rhs; return nrv; }                              \
};

#else 

#define BOOST_BINARY_OPERATOR( NAME, OP )                                     \
template <class T, class U, class B = ::boost::detail::empty_base<T> >        \
struct NAME##2 : B                                                            \
{                                                                             \
friend T operator OP( T lhs, const U& rhs ) { return lhs OP##= rhs; }       \
};                                                                            \
\
template <class T, class B = ::boost::detail::empty_base<T> >                 \
struct NAME##1 : B                                                            \
{                                                                             \
friend T operator OP( T lhs, const T& rhs ) { return lhs OP##= rhs; }       \
};

#endif 

BOOST_BINARY_OPERATOR( left_shiftable, << )
BOOST_BINARY_OPERATOR( right_shiftable, >> )

#undef BOOST_BINARY_OPERATOR

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct equivalent2 : B
{
friend bool operator==(const T& x, const U& y)
{
return !static_cast<bool>(x < y) && !static_cast<bool>(x > y);
}
};

template <class T, class B = ::boost::detail::empty_base<T> >
struct equivalent1 : B
{
friend bool operator==(const T&x, const T&y)
{
return !static_cast<bool>(x < y) && !static_cast<bool>(y < x);
}
};

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct partially_ordered2 : B
{
friend bool operator<=(const T& x, const U& y)
{ return static_cast<bool>(x < y) || static_cast<bool>(x == y); }
friend bool operator>=(const T& x, const U& y)
{ return static_cast<bool>(x > y) || static_cast<bool>(x == y); }
friend bool operator>(const U& x, const T& y)
{ return y < x; }
friend bool operator<(const U& x, const T& y)
{ return y > x; }
friend bool operator<=(const U& x, const T& y)
{ return static_cast<bool>(y > x) || static_cast<bool>(y == x); }
friend bool operator>=(const U& x, const T& y)
{ return static_cast<bool>(y < x) || static_cast<bool>(y == x); }
};

template <class T, class B = ::boost::detail::empty_base<T> >
struct partially_ordered1 : B
{
friend bool operator>(const T& x, const T& y)
{ return y < x; }
friend bool operator<=(const T& x, const T& y)
{ return static_cast<bool>(x < y) || static_cast<bool>(x == y); }
friend bool operator>=(const T& x, const T& y)
{ return static_cast<bool>(y < x) || static_cast<bool>(x == y); }
};


template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct totally_ordered2
: less_than_comparable2<T, U
, equality_comparable2<T, U, B
> > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct totally_ordered1
: less_than_comparable1<T
, equality_comparable1<T, B
> > {};

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct additive2
: addable2<T, U
, subtractable2<T, U, B
> > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct additive1
: addable1<T
, subtractable1<T, B
> > {};

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct multiplicative2
: multipliable2<T, U
, dividable2<T, U, B
> > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct multiplicative1
: multipliable1<T
, dividable1<T, B
> > {};

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct integer_multiplicative2
: multiplicative2<T, U
, modable2<T, U, B
> > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct integer_multiplicative1
: multiplicative1<T
, modable1<T, B
> > {};

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct arithmetic2
: additive2<T, U
, multiplicative2<T, U, B
> > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct arithmetic1
: additive1<T
, multiplicative1<T, B
> > {};

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct integer_arithmetic2
: additive2<T, U
, integer_multiplicative2<T, U, B
> > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct integer_arithmetic1
: additive1<T
, integer_multiplicative1<T, B
> > {};

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct bitwise2
: xorable2<T, U
, andable2<T, U
, orable2<T, U, B
> > > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct bitwise1
: xorable1<T
, andable1<T
, orable1<T, B
> > > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct unit_steppable
: incrementable<T
, decrementable<T, B
> > {};

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct shiftable2
: left_shiftable2<T, U
, right_shiftable2<T, U, B
> > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct shiftable1
: left_shiftable1<T
, right_shiftable1<T, B
> > {};

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct ring_operators2
: additive2<T, U
, subtractable2_left<T, U
, multipliable2<T, U, B
> > > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct ring_operators1
: additive1<T
, multipliable1<T, B
> > {};

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct ordered_ring_operators2
: ring_operators2<T, U
, totally_ordered2<T, U, B
> > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct ordered_ring_operators1
: ring_operators1<T
, totally_ordered1<T, B
> > {};

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct field_operators2
: ring_operators2<T, U
, dividable2<T, U
, dividable2_left<T, U, B
> > > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct field_operators1
: ring_operators1<T
, dividable1<T, B
> > {};

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct ordered_field_operators2
: field_operators2<T, U
, totally_ordered2<T, U, B
> > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct ordered_field_operators1
: field_operators1<T
, totally_ordered1<T, B
> > {};

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct euclidian_ring_operators2
: ring_operators2<T, U
, dividable2<T, U
, dividable2_left<T, U
, modable2<T, U
, modable2_left<T, U, B
> > > > > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct euclidian_ring_operators1
: ring_operators1<T
, dividable1<T
, modable1<T, B
> > > {};

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct ordered_euclidian_ring_operators2
: totally_ordered2<T, U
, euclidian_ring_operators2<T, U, B
> > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct ordered_euclidian_ring_operators1
: totally_ordered1<T
, euclidian_ring_operators1<T, B
> > {};

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct euclidean_ring_operators2
: ring_operators2<T, U
, dividable2<T, U
, dividable2_left<T, U
, modable2<T, U
, modable2_left<T, U, B
> > > > > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct euclidean_ring_operators1
: ring_operators1<T
, dividable1<T
, modable1<T, B
> > > {};

template <class T, class U, class B = ::boost::detail::empty_base<T> >
struct ordered_euclidean_ring_operators2
: totally_ordered2<T, U
, euclidean_ring_operators2<T, U, B
> > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct ordered_euclidean_ring_operators1
: totally_ordered1<T
, euclidean_ring_operators1<T, B
> > {};

template <class T, class P, class B = ::boost::detail::empty_base<T> >
struct input_iteratable
: equality_comparable1<T
, incrementable<T
, dereferenceable<T, P, B
> > > {};

template <class T, class B = ::boost::detail::empty_base<T> >
struct output_iteratable
: incrementable<T, B
> {};

template <class T, class P, class B = ::boost::detail::empty_base<T> >
struct forward_iteratable
: input_iteratable<T, P, B
> {};

template <class T, class P, class B = ::boost::detail::empty_base<T> >
struct bidirectional_iteratable
: forward_iteratable<T, P
, decrementable<T, B
> > {};

template <class T, class P, class D, class R, class B = ::boost::detail::empty_base<T> >
struct random_access_iteratable
: bidirectional_iteratable<T, P
, less_than_comparable1<T
, additive2<T, D
, indexable<T, D, R, B
> > > > {};

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} 
#endif 



#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE

# define BOOST_IMPORT_TEMPLATE4(template_name)
# define BOOST_IMPORT_TEMPLATE3(template_name)
# define BOOST_IMPORT_TEMPLATE2(template_name)
# define BOOST_IMPORT_TEMPLATE1(template_name)

#else 

#  ifndef BOOST_NO_USING_TEMPLATE

#    define BOOST_IMPORT_TEMPLATE4(template_name) using ::template_name;
#    define BOOST_IMPORT_TEMPLATE3(template_name) using ::template_name;
#    define BOOST_IMPORT_TEMPLATE2(template_name) using ::template_name;
#    define BOOST_IMPORT_TEMPLATE1(template_name) using ::template_name;

#  else

#    define BOOST_IMPORT_TEMPLATE4(template_name)                                             \
template <class T, class U, class V, class W, class B = ::boost::detail::empty_base<T> > \
struct template_name : ::template_name<T, U, V, W, B> {};

#    define BOOST_IMPORT_TEMPLATE3(template_name)                                    \
template <class T, class U, class V, class B = ::boost::detail::empty_base<T> > \
struct template_name : ::template_name<T, U, V, B> {};

#    define BOOST_IMPORT_TEMPLATE2(template_name)                           \
template <class T, class U, class B = ::boost::detail::empty_base<T> > \
struct template_name : ::template_name<T, U, B> {};

#    define BOOST_IMPORT_TEMPLATE1(template_name)                  \
template <class T, class B = ::boost::detail::empty_base<T> > \
struct template_name : ::template_name<T, B> {};

#  endif 

#endif 



namespace boost {
namespace detail {
struct true_t {};
struct false_t {};
} 

template<class T> struct is_chained_base {
typedef ::boost::detail::false_t value;
};

} 

# define BOOST_OPERATOR_TEMPLATE4(template_name4)                     \
BOOST_IMPORT_TEMPLATE4(template_name4)                              \
template<class T, class U, class V, class W, class B>               \
struct is_chained_base< ::boost::template_name4<T, U, V, W, B> > {  \
typedef ::boost::detail::true_t value;                            \
};

# define BOOST_OPERATOR_TEMPLATE3(template_name3)                     \
BOOST_IMPORT_TEMPLATE3(template_name3)                              \
template<class T, class U, class V, class B>                        \
struct is_chained_base< ::boost::template_name3<T, U, V, B> > {     \
typedef ::boost::detail::true_t value;                            \
};

# define BOOST_OPERATOR_TEMPLATE2(template_name2)                  \
BOOST_IMPORT_TEMPLATE2(template_name2)                           \
template<class T, class U, class B>                              \
struct is_chained_base< ::boost::template_name2<T, U, B> > {     \
typedef ::boost::detail::true_t value;                         \
};

# define BOOST_OPERATOR_TEMPLATE1(template_name1)                  \
BOOST_IMPORT_TEMPLATE1(template_name1)                           \
template<class T, class B>                                       \
struct is_chained_base< ::boost::template_name1<T, B> > {        \
typedef ::boost::detail::true_t value;                         \
};


# define BOOST_OPERATOR_TEMPLATE(template_name)                    \
template <class T                                                  \
,class U = T                                              \
,class B = ::boost::detail::empty_base<T>                 \
,class O = typename is_chained_base<U>::value             \
>                                                         \
struct template_name : template_name##2<T, U, B> {};               \
\
template<class T, class U, class B>                                \
struct template_name<T, U, B, ::boost::detail::true_t>             \
: template_name##1<T, U> {};                                     \
\
template <class T, class B>                                        \
struct template_name<T, T, B, ::boost::detail::false_t>            \
: template_name##1<T, B> {};                                     \
\
template<class T, class U, class B, class O>                       \
struct is_chained_base< ::boost::template_name<T, U, B, O> > {     \
typedef ::boost::detail::true_t value;                           \
};                                                                 \
\
BOOST_OPERATOR_TEMPLATE2(template_name##2)                         \
BOOST_OPERATOR_TEMPLATE1(template_name##1)



namespace boost {

BOOST_OPERATOR_TEMPLATE(less_than_comparable)
BOOST_OPERATOR_TEMPLATE(equality_comparable)
BOOST_OPERATOR_TEMPLATE(multipliable)
BOOST_OPERATOR_TEMPLATE(addable)
BOOST_OPERATOR_TEMPLATE(subtractable)
BOOST_OPERATOR_TEMPLATE2(subtractable2_left)
BOOST_OPERATOR_TEMPLATE(dividable)
BOOST_OPERATOR_TEMPLATE2(dividable2_left)
BOOST_OPERATOR_TEMPLATE(modable)
BOOST_OPERATOR_TEMPLATE2(modable2_left)
BOOST_OPERATOR_TEMPLATE(xorable)
BOOST_OPERATOR_TEMPLATE(andable)
BOOST_OPERATOR_TEMPLATE(orable)

BOOST_OPERATOR_TEMPLATE1(incrementable)
BOOST_OPERATOR_TEMPLATE1(decrementable)

BOOST_OPERATOR_TEMPLATE2(dereferenceable)
BOOST_OPERATOR_TEMPLATE3(indexable)

BOOST_OPERATOR_TEMPLATE(left_shiftable)
BOOST_OPERATOR_TEMPLATE(right_shiftable)
BOOST_OPERATOR_TEMPLATE(equivalent)
BOOST_OPERATOR_TEMPLATE(partially_ordered)

BOOST_OPERATOR_TEMPLATE(totally_ordered)
BOOST_OPERATOR_TEMPLATE(additive)
BOOST_OPERATOR_TEMPLATE(multiplicative)
BOOST_OPERATOR_TEMPLATE(integer_multiplicative)
BOOST_OPERATOR_TEMPLATE(arithmetic)
BOOST_OPERATOR_TEMPLATE(integer_arithmetic)
BOOST_OPERATOR_TEMPLATE(bitwise)
BOOST_OPERATOR_TEMPLATE1(unit_steppable)
BOOST_OPERATOR_TEMPLATE(shiftable)
BOOST_OPERATOR_TEMPLATE(ring_operators)
BOOST_OPERATOR_TEMPLATE(ordered_ring_operators)
BOOST_OPERATOR_TEMPLATE(field_operators)
BOOST_OPERATOR_TEMPLATE(ordered_field_operators)
BOOST_OPERATOR_TEMPLATE(euclidian_ring_operators)
BOOST_OPERATOR_TEMPLATE(ordered_euclidian_ring_operators)
BOOST_OPERATOR_TEMPLATE(euclidean_ring_operators)
BOOST_OPERATOR_TEMPLATE(ordered_euclidean_ring_operators)
BOOST_OPERATOR_TEMPLATE2(input_iteratable)
BOOST_OPERATOR_TEMPLATE1(output_iteratable)
BOOST_OPERATOR_TEMPLATE2(forward_iteratable)
BOOST_OPERATOR_TEMPLATE2(bidirectional_iteratable)
BOOST_OPERATOR_TEMPLATE4(random_access_iteratable)

#undef BOOST_OPERATOR_TEMPLATE
#undef BOOST_OPERATOR_TEMPLATE4
#undef BOOST_OPERATOR_TEMPLATE3
#undef BOOST_OPERATOR_TEMPLATE2
#undef BOOST_OPERATOR_TEMPLATE1
#undef BOOST_IMPORT_TEMPLATE1
#undef BOOST_IMPORT_TEMPLATE2
#undef BOOST_IMPORT_TEMPLATE3
#undef BOOST_IMPORT_TEMPLATE4

template <class T, class U>
struct operators2
: totally_ordered2<T,U
, integer_arithmetic2<T,U
, bitwise2<T,U
> > > {};

template <class T, class U = T>
struct operators : operators2<T, U> {};

template <class T> struct operators<T, T>
: totally_ordered<T
, integer_arithmetic<T
, bitwise<T
, unit_steppable<T
> > > > {};

template <class T,
class V,
class D = std::ptrdiff_t,
class P = V const *,
class R = V const &>
struct input_iterator_helper
: input_iteratable<T, P
, std::iterator<std::input_iterator_tag, V, D, P, R
> > {};

template<class T>
struct output_iterator_helper
: output_iteratable<T
, std::iterator<std::output_iterator_tag, void, void, void, void
> >
{
T& operator*()  { return static_cast<T&>(*this); }
T& operator++() { return static_cast<T&>(*this); }
};

template <class T,
class V,
class D = std::ptrdiff_t,
class P = V*,
class R = V&>
struct forward_iterator_helper
: forward_iteratable<T, P
, std::iterator<std::forward_iterator_tag, V, D, P, R
> > {};

template <class T,
class V,
class D = std::ptrdiff_t,
class P = V*,
class R = V&>
struct bidirectional_iterator_helper
: bidirectional_iteratable<T, P
, std::iterator<std::bidirectional_iterator_tag, V, D, P, R
> > {};

template <class T,
class V, 
class D = std::ptrdiff_t,
class P = V*,
class R = V&>
struct random_access_iterator_helper
: random_access_iteratable<T, P, D, R
, std::iterator<std::random_access_iterator_tag, V, D, P, R
> >
{
friend D requires_difference_operator(const T& x, const T& y) {
return x - y;
}
}; 

} 

#if defined(__sgi) && !defined(__GNUC__)
#pragma reset woff 1234
#endif

#endif 
