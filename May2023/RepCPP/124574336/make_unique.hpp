
#ifndef BOOST_MOVE_MAKE_UNIQUE_HPP_INCLUDED
#define BOOST_MOVE_MAKE_UNIQUE_HPP_INCLUDED

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/move/detail/config_begin.hpp>
#include <boost/move/detail/workaround.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/move/unique_ptr.hpp>
#include <cstddef>   
#include <boost/move/detail/unique_ptr_meta_utils.hpp>
#ifdef BOOST_NO_CXX11_VARIADIC_TEMPLATES
#  include <boost/move/detail/fwd_macros.hpp>
#endif


#if !defined(BOOST_MOVE_DOXYGEN_INVOKED)

namespace std {   

struct nothrow_t;

}  

namespace boost{
namespace move_upmu {

template<class T>
struct unique_ptr_if
{
typedef ::boost::movelib::unique_ptr<T> t_is_not_array;
};

template<class T>
struct unique_ptr_if<T[]>
{
typedef ::boost::movelib::unique_ptr<T[]> t_is_array_of_unknown_bound;
};

template<class T, std::size_t N>
struct unique_ptr_if<T[N]>
{
typedef void t_is_array_of_known_bound;
};

template <int Dummy = 0>
struct nothrow_holder
{
static std::nothrow_t *pnothrow;   
};

template <int Dummy>
std::nothrow_t *nothrow_holder<Dummy>::pnothrow = 
reinterpret_cast<std::nothrow_t *>(0x1234);  

}  
}  

#endif   

namespace boost{
namespace movelib {

#if defined(BOOST_MOVE_DOXYGEN_INVOKED) || !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES)

template<class T, class... Args>
inline BOOST_MOVE_DOC1ST(unique_ptr<T>, 
typename ::boost::move_upmu::unique_ptr_if<T>::t_is_not_array)
make_unique(BOOST_FWD_REF(Args)... args)
{  return unique_ptr<T>(new T(::boost::forward<Args>(args)...));  }

template<class T, class... Args>
inline BOOST_MOVE_DOC1ST(unique_ptr<T>, 
typename ::boost::move_upmu::unique_ptr_if<T>::t_is_not_array)
make_unique_nothrow(BOOST_FWD_REF(Args)... args)
{  return unique_ptr<T>(new (*boost::move_upmu::nothrow_holder<>::pnothrow)T(::boost::forward<Args>(args)...));  }

#else
#define BOOST_MOVE_MAKE_UNIQUE_CODE(N)\
template<class T BOOST_MOVE_I##N BOOST_MOVE_CLASS##N>\
typename ::boost::move_upmu::unique_ptr_if<T>::t_is_not_array\
make_unique( BOOST_MOVE_UREF##N)\
{  return unique_ptr<T>( new T( BOOST_MOVE_FWD##N ) );  }\
\
template<class T BOOST_MOVE_I##N BOOST_MOVE_CLASS##N>\
typename ::boost::move_upmu::unique_ptr_if<T>::t_is_not_array\
make_unique_nothrow( BOOST_MOVE_UREF##N)\
{  return unique_ptr<T>( new (*boost::move_upmu::nothrow_holder<>::pnothrow)T ( BOOST_MOVE_FWD##N ) );  }\
BOOST_MOVE_ITERATE_0TO9(BOOST_MOVE_MAKE_UNIQUE_CODE)
#undef BOOST_MOVE_MAKE_UNIQUE_CODE

#endif

template<class T>
inline BOOST_MOVE_DOC1ST(unique_ptr<T>, 
typename ::boost::move_upmu::unique_ptr_if<T>::t_is_not_array)
make_unique_definit()
{
return unique_ptr<T>(new T);
}

template<class T>
inline BOOST_MOVE_DOC1ST(unique_ptr<T>, 
typename ::boost::move_upmu::unique_ptr_if<T>::t_is_not_array)
make_unique_nothrow_definit()
{
return unique_ptr<T>(new (*boost::move_upmu::nothrow_holder<>::pnothrow)T);
}

template<class T>
inline BOOST_MOVE_DOC1ST(unique_ptr<T>, 
typename ::boost::move_upmu::unique_ptr_if<T>::t_is_array_of_unknown_bound)
make_unique(std::size_t n)
{
typedef typename ::boost::move_upmu::remove_extent<T>::type U;
return unique_ptr<T>(new U[n]());
}

template<class T>
inline BOOST_MOVE_DOC1ST(unique_ptr<T>, 
typename ::boost::move_upmu::unique_ptr_if<T>::t_is_array_of_unknown_bound)
make_unique_nothrow(std::size_t n)
{
typedef typename ::boost::move_upmu::remove_extent<T>::type U;
return unique_ptr<T>(new (*boost::move_upmu::nothrow_holder<>::pnothrow)U[n]());
}

template<class T>
inline BOOST_MOVE_DOC1ST(unique_ptr<T>, 
typename ::boost::move_upmu::unique_ptr_if<T>::t_is_array_of_unknown_bound)
make_unique_definit(std::size_t n)
{
typedef typename ::boost::move_upmu::remove_extent<T>::type U;
return unique_ptr<T>(new U[n]);
}

template<class T>
inline BOOST_MOVE_DOC1ST(unique_ptr<T>, 
typename ::boost::move_upmu::unique_ptr_if<T>::t_is_array_of_unknown_bound)
make_unique_nothrow_definit(std::size_t n)
{
typedef typename ::boost::move_upmu::remove_extent<T>::type U;
return unique_ptr<T>(new (*boost::move_upmu::nothrow_holder<>::pnothrow) U[n]);
}

#if !defined(BOOST_NO_CXX11_DELETED_FUNCTIONS)

template<class T, class... Args>
inline BOOST_MOVE_DOC1ST(unspecified, 
typename ::boost::move_upmu::unique_ptr_if<T>::t_is_array_of_known_bound)
make_unique(BOOST_FWD_REF(Args) ...) = delete;

template<class T, class... Args>
inline BOOST_MOVE_DOC1ST(unspecified, 
typename ::boost::move_upmu::unique_ptr_if<T>::t_is_array_of_known_bound)
make_unique_definit(BOOST_FWD_REF(Args) ...) = delete;

template<class T, class... Args>
inline BOOST_MOVE_DOC1ST(unspecified, 
typename ::boost::move_upmu::unique_ptr_if<T>::t_is_array_of_known_bound)
make_unique_nothrow(BOOST_FWD_REF(Args) ...) = delete;

template<class T, class... Args>
inline BOOST_MOVE_DOC1ST(unspecified, 
typename ::boost::move_upmu::unique_ptr_if<T>::t_is_array_of_known_bound)
make_unique_nothrow_definit(BOOST_FWD_REF(Args) ...) = delete;

#endif

}  

}  

#include <boost/move/detail/config_end.hpp>

#endif   
