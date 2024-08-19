#ifndef BOOST_GIL_EXTENSION_IO_JPEG_DETAIL_SCANLINE_READ_HPP
#define BOOST_GIL_EXTENSION_IO_JPEG_DETAIL_SCANLINE_READ_HPP


#include <boost/gil/extension/io/jpeg/detail/base.hpp>
#include <boost/gil/extension/io/jpeg/detail/is_allowed.hpp>
#include <boost/gil/extension/io/jpeg/detail/reader_backend.hpp>

#include <boost/gil/io/base.hpp>
#include <boost/gil/io/conversion_policies.hpp>
#include <boost/gil/io/device.hpp>
#include <boost/gil/io/reader_base.hpp>
#include <boost/gil/io/scanline_read_iterator.hpp>
#include <boost/gil/io/typedefs.hpp>

#include <csetjmp>
#include <vector>

namespace boost { namespace gil {

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(push)
#pragma warning(disable:4611) 
#endif

template< typename Device >
class scanline_reader< Device
, jpeg_tag
>
: public reader_backend< Device
, jpeg_tag
>
{
public:

using tag_t = jpeg_tag;
using backend_t = reader_backend<Device, tag_t>;
using this_t = scanline_reader<Device, tag_t>;
using iterator_t = scanline_read_iterator<this_t>;

public:
scanline_reader( Device&                                device
, const image_read_settings< jpeg_tag >& settings
)
: reader_backend< Device
, jpeg_tag
>( device
, settings
)
{
initialize();
}

void read( byte_t* dst
, int
)
{
if( setjmp( this->_mark )) { this->raise_error(); }

read_scanline( dst );
}

void skip( byte_t* dst, int )
{
if( setjmp( this->_mark )) { this->raise_error(); }

read_scanline( dst );
}

iterator_t begin() { return iterator_t( *this ); }
iterator_t end()   { return iterator_t( *this, this->_info._height ); }

private:

void initialize()
{
this->get()->dct_method = this->_settings._dct_method;

io_error_if( jpeg_start_decompress( this->get() ) == false
, "Cannot start decompression." );

switch( this->_info._color_space )
{
case JCS_GRAYSCALE:
{
this->_scanline_length = this->_info._width;

break;
}

case JCS_RGB:
case JCS_YCbCr:
{
this->_scanline_length = this->_info._width * num_channels< rgb8_view_t >::value;

break;
}


case JCS_CMYK:
case JCS_YCCK:
{
this->get()->out_color_space = JCS_CMYK;
this->_scanline_length = this->_info._width * num_channels< cmyk8_view_t >::value;

break;
}

default: { io_error( "Unsupported jpeg color space." ); }
}
}

void read_scanline( byte_t* dst )
{
JSAMPLE *row_adr = reinterpret_cast< JSAMPLE* >( dst );

io_error_if( jpeg_read_scanlines( this->get()
, &row_adr
, 1
) != 1
, "jpeg_read_scanlines: fail to read JPEG file"
);

}
};

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(pop)
#endif

} 
} 

#endif
