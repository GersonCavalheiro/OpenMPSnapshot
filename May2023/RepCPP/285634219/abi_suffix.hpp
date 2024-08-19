



#ifndef BOOST_CONFIG_ABI_PREFIX_HPP
# error Header boost/config/abi_suffix.hpp must only be used after boost/config/abi_prefix.hpp
#else
# undef BOOST_CONFIG_ABI_PREFIX_HPP
#endif

#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_SUFFIX
#endif

#if defined( __BORLANDC__ )
#pragma nopushoptwarn
#endif


