

#ifdef BOOST_MSVC
# pragma warning(push)
# pragma warning(disable: 4511) 
# pragma warning(disable: 4512) 
# pragma warning(disable: 4100) 
# pragma warning(disable: 4996) 
# pragma warning(disable: 4355) 
# pragma warning(disable: 4706) 
# pragma warning(disable: 4251) 
# pragma warning(disable: 4127) 
# pragma warning(disable: 4290) 
# pragma warning(disable: 4180) 
# pragma warning(disable: 4275) 
# pragma warning(disable: 4267) 
# pragma warning(disable: 4511) 
#endif

#if defined(BOOST_CLANG) && (BOOST_CLANG == 1)
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wvariadic-macros"
# pragma clang diagnostic ignored "-Wmissing-declarations"
#endif

#if defined(BOOST_GCC) && (BOOST_GCC >= 4 * 10000 + 6 * 100)
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wvariadic-macros"
# pragma GCC diagnostic ignored "-Wmissing-declarations"
#endif

