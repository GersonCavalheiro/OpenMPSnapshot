

#ifndef BOOST_BIMAP_CONTAINER_ADAPTOR_DETAIL_NON_UNIQUE_CONTAINER_HELPER_HPP
#define BOOST_BIMAP_CONTAINER_ADAPTOR_DETAIL_NON_UNIQUE_CONTAINER_HELPER_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>


#define BOOST_BIMAP_NON_UNIQUE_CONTAINER_ADAPTOR_INSERT_FUNCTIONS             \
\
template <class InputIterator>                                                \
void insert(InputIterator iterBegin, InputIterator iterEnd)                   \
{                                                                             \
for( ; iterBegin != iterEnd ; ++iterBegin )                               \
{                                                                         \
this->base().insert(                                                  \
this->template functor<                                           \
BOOST_DEDUCED_TYPENAME base_::value_to_base>()(               \
BOOST_DEDUCED_TYPENAME base_::value_type(*iterBegin)) );  \
}                                                                         \
}                                                                             \
\
BOOST_DEDUCED_TYPENAME base_::iterator insert(                                \
BOOST_DEDUCED_TYPENAME ::boost::call_traits<                              \
BOOST_DEDUCED_TYPENAME base_::value_type >::param_type x)             \
{                                                                             \
return this->base().insert( this->template functor<                       \
BOOST_DEDUCED_TYPENAME base_::             \
value_to_base>()(x) );                \
}                                                                             \
\
BOOST_DEDUCED_TYPENAME base_::iterator                                        \
insert(BOOST_DEDUCED_TYPENAME base_::iterator pos,                        \
BOOST_DEDUCED_TYPENAME ::boost::call_traits<                   \
BOOST_DEDUCED_TYPENAME base_::value_type >::param_type x) \
{                                                                             \
return this->template functor<                                            \
BOOST_DEDUCED_TYPENAME base_::iterator_from_base>()(                  \
this->base().insert(this->template functor<                       \
BOOST_DEDUCED_TYPENAME base_::iterator_to_base>()(pos),       \
this->template functor<                                           \
BOOST_DEDUCED_TYPENAME base_::value_to_base>()(x))            \
);                                                                        \
}


#endif 


