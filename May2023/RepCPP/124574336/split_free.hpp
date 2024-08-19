#ifndef BOOST_SERIALIZATION_SPLIT_FREE_HPP
#define BOOST_SERIALIZATION_SPLIT_FREE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/serialization/serialization.hpp>

namespace boost {
namespace archive {
namespace detail {
template<class Archive> class interface_oarchive;
template<class Archive> class interface_iarchive;
} 
} 

namespace serialization {

template<class Archive, class T>
struct free_saver {
static void invoke(
Archive & ar,
const  T & t,
const unsigned int file_version
){
const version_type v(file_version);
save(ar, t, v);
}
};
template<class Archive, class T>
struct free_loader {
static void invoke(
Archive & ar,
T & t,
const unsigned int file_version
){
const version_type v(file_version);
load(ar, t, v);
}
};

template<class Archive, class T>
inline void split_free(
Archive & ar,
T & t,
const unsigned int file_version
){
typedef typename mpl::eval_if<
typename Archive::is_saving,
mpl::identity< free_saver<Archive, T> >,
mpl::identity< free_loader<Archive, T> >
>::type typex;
typex::invoke(ar, t, file_version);
}

} 
} 

#define BOOST_SERIALIZATION_SPLIT_FREE(T)       \
namespace boost { namespace serialization {     \
template<class Archive>                         \
inline void serialize(                          \
Archive & ar,                               \
T & t,                                      \
const unsigned int file_version             \
){                                              \
split_free(ar, t, file_version);            \
}                                               \
}}


#endif 
