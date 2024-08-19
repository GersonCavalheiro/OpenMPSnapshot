



#ifndef BOOST_POLYMORPHIC_CAST_HPP
#define BOOST_POLYMORPHIC_CAST_HPP

# include <boost/config.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#   pragma once
#endif

# include <boost/assert.hpp>
# include <boost/core/addressof.hpp>
# include <boost/core/enable_if.hpp>
# include <boost/throw_exception.hpp>
# include <boost/type_traits/is_reference.hpp> 
# include <boost/type_traits/remove_reference.hpp>

# include <typeinfo>

namespace boost
{



template <class Target, class Source>
inline Target polymorphic_cast(Source* x)
{
Target tmp = dynamic_cast<Target>(x);
if ( tmp == 0 ) boost::throw_exception( std::bad_cast() );
return tmp;
}





template <class Target, class Source>
inline Target polymorphic_downcast(Source* x)
{
BOOST_ASSERT( dynamic_cast<Target>(x) == x );  
return static_cast<Target>(x);
}




template <class Target, class Source>
inline typename boost::enable_if_c<
boost::is_reference<Target>::value, Target
>::type polymorphic_downcast(Source& x)
{
typedef typename boost::remove_reference<Target>::type* target_pointer_type;
return *boost::polymorphic_downcast<target_pointer_type>(
boost::addressof(x)
);
}

} 

#endif  
