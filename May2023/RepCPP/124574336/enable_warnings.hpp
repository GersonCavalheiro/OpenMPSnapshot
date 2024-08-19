

#ifdef BOOST_MSVC
# pragma warning(default: 4511) 
# pragma warning(default: 4512) 
# pragma warning(default: 4100) 
# pragma warning(default: 4996) 
# pragma warning(default: 4355) 
# pragma warning(default: 4706) 
# pragma warning(default: 4251) 
# pragma warning(default: 4127) 
# pragma warning(default: 4290) 
# pragma warning(default: 4180) 
# pragma warning(default: 4275) 
# pragma warning(default: 4267) 
# pragma warning(default: 4511) 
# pragma warning(pop)
#endif

#if defined(BOOST_CLANG) && (BOOST_CLANG == 1)
#pragma clang diagnostic pop
#endif

#if defined(BOOST_GCC) && (BOOST_GCC >= 4 * 10000 + 6 * 100)
# pragma GCC diagnostic pop
#endif

