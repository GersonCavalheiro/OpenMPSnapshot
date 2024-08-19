#ifndef BOOST_SERIALIZATION_SPLIT_MEMBER_HPP
#define BOOST_SERIALIZATION_SPLIT_MEMBER_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>

#include <boost/serialization/access.hpp>

namespace boost {
namespace archive {
namespace detail {
template<class Archive> class interface_oarchive;
template<class Archive> class interface_iarchive;
} 
} 

namespace serialization {
namespace detail {

template<class Archive, class T>
struct member_saver {
static void invoke(
Archive & ar,
const T & t,
const unsigned int file_version
){
access::member_save(ar, t, file_version);
}
};

template<class Archive, class T>
struct member_loader {
static void invoke(
Archive & ar,
T & t,
const unsigned int file_version
){
access::member_load(ar, t, file_version);
}
};

} 

template<class Archive, class T>
inline void split_member(
Archive & ar, T & t, const unsigned int file_version
){
typedef typename mpl::eval_if<
typename Archive::is_saving,
mpl::identity<detail::member_saver<Archive, T> >,
mpl::identity<detail::member_loader<Archive, T> >
>::type typex;
typex::invoke(ar, t, file_version);
}

} 
} 

#define BOOST_SERIALIZATION_SPLIT_MEMBER()                       \
template<class Archive>                                          \
void serialize(                                                  \
Archive &ar,                                                 \
const unsigned int file_version                              \
){                                                               \
boost::serialization::split_member(ar, *this, file_version); \
}                                                                \


#endif 
