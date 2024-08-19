

#include <boost/config.hpp>

#if defined(BOOST_GCC)&&(BOOST_GCC>=4*10000+6*100)
#if !defined(BOOST_MULTI_INDEX_DETAIL_RESTORE_WSTRICT_ALIASING)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#else
#pragma GCC diagnostic pop
#endif
#endif
