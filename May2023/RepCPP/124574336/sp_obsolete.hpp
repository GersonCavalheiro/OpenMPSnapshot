#ifndef BOOST_SMART_PTR_DETAIL_SP_OBSOLETE_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_OBSOLETE_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif



#include <boost/config/pragma_message.hpp>

#if !defined( BOOST_SP_NO_OBSOLETE_MESSAGE )

#define BOOST_SP_OBSOLETE() BOOST_PRAGMA_MESSAGE("This platform-specific implementation is presumed obsolete and is slated for removal. If you want it retained, please open an issue in https:

#else

#define BOOST_SP_OBSOLETE()

#endif

#endif 
