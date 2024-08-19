
#if !defined(BOOST_SPIRIT_X3_CAST_CHAR_NOVEMBER_10_2006_0907AM)
#define BOOST_SPIRIT_X3_CAST_CHAR_NOVEMBER_10_2006_0907AM

#include <boost/type_traits/is_signed.hpp>
#include <boost/type_traits/make_unsigned.hpp>
#include <boost/type_traits/make_signed.hpp>

namespace boost { namespace spirit { namespace x3 { namespace detail
{

template <typename TargetChar, typename SourceChar>
TargetChar cast_char(SourceChar ch)
{
#if defined(_MSC_VER)
# pragma warning(push)
# pragma warning(disable: 4127) 
#endif
if (is_signed<TargetChar>::value != is_signed<SourceChar>::value)
{
if (is_signed<SourceChar>::value)
{
typedef typename make_unsigned<SourceChar>::type USourceChar;
return TargetChar(USourceChar(ch));
}
else
{
typedef typename make_signed<SourceChar>::type SSourceChar;
return TargetChar(SSourceChar(ch));
}
}
else
{
return TargetChar(ch); 
}
#if defined(_MSC_VER)
# pragma warning(pop)
#endif
}
}}}}

#endif


