
#ifndef BOOST_SPIRIT_ACTOR_REF_VALUE_ACTOR_HPP
#define BOOST_SPIRIT_ACTOR_REF_VALUE_ACTOR_HPP

#include <boost/detail/workaround.hpp>

#include <boost/spirit/home/classic/namespace.hpp>

namespace boost { namespace spirit {

BOOST_SPIRIT_CLASSIC_NAMESPACE_BEGIN

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(push)
#pragma warning(disable:4512) 
#endif

template<
typename T,
typename ActionT
>
class ref_value_actor : public ActionT
{
private:
T& ref;
public:
explicit
ref_value_actor(T& ref_)
: ref(ref_){}


template<typename T2>
void operator()(T2 const& val_) const
{
this->act(ref,val_); 
}


template<typename IteratorT>
void operator()(
IteratorT const& first_,
IteratorT const& last_
) const
{
this->act(ref,first_,last_); 
}
};

BOOST_SPIRIT_CLASSIC_NAMESPACE_END

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(pop)
#endif

}}

#endif
