


#if (defined _MSC_VER) && (_MSC_VER == 1200)
# pragma warning (disable : 4786) 
#endif

#if ! ((defined _MSC_VER) && (_MSC_VER <= 1300)) 

#include <boost/config.hpp>

#if defined(__SGI_STL_PORT) && (__SGI_STL_PORT < 0x500)

#include <boost/archive/codecvt_null.hpp>


namespace std {

template
locale::locale(
const locale& __loc, boost::archive::codecvt_null<char> * __f
);

template
locale::locale(
const locale& __loc, boost::archive::codecvt_null<wchar_t> * __f
);

} 

#endif

#endif
