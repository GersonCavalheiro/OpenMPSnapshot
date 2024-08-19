
#ifndef BOOST_PROTO_LITERAL_HPP_EAN_01_03_2007
#define BOOST_PROTO_LITERAL_HPP_EAN_01_03_2007

#include <boost/config.hpp>
#include <boost/proto/proto_fwd.hpp>
#include <boost/proto/expr.hpp>
#include <boost/proto/traits.hpp>
#include <boost/proto/extends.hpp>

namespace boost { namespace proto
{
namespace utility
{
template<
typename T
, typename Domain 
>
struct literal
: extends<basic_expr<tag::terminal, term<T>, 0>, literal<T, Domain>, Domain>
{
private:
typedef basic_expr<tag::terminal, term<T>, 0> terminal_type;
typedef extends<terminal_type, literal<T, Domain>, Domain> base_type;
typedef literal<T, Domain> literal_t;

public:
typedef typename detail::term_traits<T>::value_type       value_type;
typedef typename detail::term_traits<T>::reference        reference;
typedef typename detail::term_traits<T>::const_reference  const_reference;

literal()
: base_type(terminal_type::make(T()))
{}

#ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
literal(literal const &) = default;
#endif

template<typename U>
literal(U &u)
: base_type(terminal_type::make(u))
{}

template<typename U>
literal(U const &u)
: base_type(terminal_type::make(u))
{}

template<typename U>
literal(literal<U, Domain> const &u)
: base_type(terminal_type::make(u.get()))
{}

BOOST_PROTO_EXTENDS_USING_ASSIGN(literal_t)

reference get()
{
return proto::value(*this);
}

const_reference get() const
{
return proto::value(*this);
}
};
}

template<typename T>
inline literal<T &> const lit(T &t)
{
return literal<T &>(t);
}

template<typename T>
inline literal<T const &> const lit(T const &t)
{
#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4180) 
#endif

return literal<T const &>(t);

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif
}

}}

#endif
