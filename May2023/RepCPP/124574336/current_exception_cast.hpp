

#ifndef BOOST_EXCEPTION_7E83C166200811DE885E826156D89593
#define BOOST_EXCEPTION_7E83C166200811DE885E826156D89593

#ifndef BOOST_EXCEPTION_ENABLE_WARNINGS
#if __GNUC__*100+__GNUC_MINOR__>301
#pragma GCC system_header
#endif
#ifdef __clang__
#pragma clang system_header
#endif
#ifdef _MSC_VER
#pragma warning(push,1)
#endif
#endif

namespace
boost
{
template <class E>
inline
E *
current_exception_cast()
{
try
{
throw;
}
catch(
E & e )
{
return &e;
}
catch(
...)
{
return 0;
}
}
}

#if defined(_MSC_VER) && !defined(BOOST_EXCEPTION_ENABLE_WARNINGS)
#pragma warning(pop)
#endif
#endif
