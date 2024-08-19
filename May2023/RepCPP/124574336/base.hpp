#ifndef BOOST_GIL_EXTENSION_IO_JPEG_DETAIL_BASE_HPP
#define BOOST_GIL_EXTENSION_IO_JPEG_DETAIL_BASE_HPP

#include <boost/gil/extension/io/jpeg/tags.hpp>

#include <csetjmp>

namespace boost { namespace gil {

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(push)
#pragma warning(disable:4324) 
#endif

class jpeg_io_base
{

protected:

jpeg_error_mgr _jerr;
jmp_buf        _mark;
};

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(pop)
#endif

} 
} 

#endif
