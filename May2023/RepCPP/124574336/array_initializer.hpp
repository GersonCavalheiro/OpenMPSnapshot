
#ifndef BOOST_INTRUSIVE_DETAIL_ARRAY_INITIALIZER_HPP
#define BOOST_INTRUSIVE_DETAIL_ARRAY_INITIALIZER_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/config.hpp>
#include <boost/core/no_exceptions_support.hpp>
#include <boost/move/detail/placement_new.hpp>

namespace boost {
namespace intrusive {
namespace detail {

union max_align
{
char        char_;
short       short_;
int         int_;
long        long_;
#ifdef BOOST_HAS_LONG_LONG
::boost::long_long_type  long_long_;
#endif
float       float_;
double      double_;
long double long_double_;
void *      void_ptr_;
};

template<class T, std::size_t N>
class array_initializer
{
public:
template<class CommonInitializer>
array_initializer(const CommonInitializer &init)
{
char *init_buf = (char*)rawbuf;
std::size_t i = 0;
BOOST_TRY{
for(; i != N; ++i){
::new(init_buf, boost_move_new_t()) T(init);
init_buf += sizeof(T);
}
}
BOOST_CATCH(...){
while(i--){
init_buf -= sizeof(T);
((T*)init_buf)->~T();
}
BOOST_RETHROW;
}
BOOST_CATCH_END
}

operator T* ()
{  return (T*)(rawbuf);  }

operator const T*() const
{  return (const T*)(rawbuf);  }

~array_initializer()
{
char *init_buf = (char*)rawbuf + N*sizeof(T);
for(std::size_t i = 0; i != N; ++i){
init_buf -= sizeof(T);
((T*)init_buf)->~T();
}
}

private:
detail::max_align rawbuf[(N*sizeof(T)-1)/sizeof(detail::max_align)+1];
};

}  
}  
}  

#endif 
