
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_ADAPTOR_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_ADAPTOR_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/ref.hpp>
#include <boost/implicit_cast.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/dynamic/matchable.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename Xpr, typename Base>
struct xpression_adaptor
: Base 
{
typedef typename Base::iterator_type iterator_type;
typedef typename iterator_value<iterator_type>::type char_type;

Xpr xpr_;

xpression_adaptor(Xpr const &xpr)
#if BOOST_WORKAROUND(__GNUC__, BOOST_TESTED_AT(4))
__attribute__((__noinline__))
#endif
: xpr_(xpr)
{
}

virtual bool match(match_state<iterator_type> &state) const
{
typedef typename boost::unwrap_reference<Xpr const>::type xpr_type;
return implicit_cast<xpr_type &>(this->xpr_).match(state);
}

void link(xpression_linker<char_type> &linker) const
{
this->xpr_.link(linker);
}

void peek(xpression_peeker<char_type> &peeker) const
{
this->xpr_.peek(peeker);
}

private:
xpression_adaptor &operator =(xpression_adaptor const &);
};

template<typename Base, typename Xpr>
inline intrusive_ptr<Base const> make_adaptor(Xpr const &xpr)
{
return intrusive_ptr<Base const>(new xpression_adaptor<Xpr, Base>(xpr));
}

}}} 

#endif
