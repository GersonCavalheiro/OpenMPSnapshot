#ifndef BOOST_GIL_EXTENSION_IO_TIFF_DETAIL_WRITER_BACKEND_HPP
#define BOOST_GIL_EXTENSION_IO_TIFF_DETAIL_WRITER_BACKEND_HPP

#include <boost/gil/extension/io/tiff/tags.hpp>
#include <boost/gil/extension/io/tiff/detail/device.hpp>

#include <boost/gil/detail/mp11.hpp>

namespace boost { namespace gil {

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(push)
#pragma warning(disable:4512) 
#endif

template< typename Device >
struct writer_backend< Device
, tiff_tag
>
{
public:

using format_tag_t = tiff_tag;

public:

writer_backend( const Device&                       io_dev
, const image_write_info< tiff_tag >& info
)
: _io_dev( io_dev )
, _info( info )
{}

protected:

template< typename View >
void write_header( const View& view )
{
using pixel_t = typename View::value_type;

using channel_t = typename channel_traits<typename element_type<pixel_t>::type>::value_type;
using color_space_t = typename color_space_type<View>::type;

if(! this->_info._photometric_interpretation_user_defined )
{
this->_info._photometric_interpretation = detail::photometric_interpretation< color_space_t >::value;
}

tiff_image_width::type  width  = (tiff_image_width::type)  view.width();
tiff_image_height::type height = (tiff_image_height::type) view.height();

this->_io_dev.template set_property< tiff_image_width  >( width  );
this->_io_dev.template set_property< tiff_image_height >( height );

this->_io_dev.template set_property<tiff_planar_configuration>( this->_info._planar_configuration );

tiff_samples_per_pixel::type samples_per_pixel = num_channels< pixel_t >::value;
this->_io_dev.template set_property<tiff_samples_per_pixel>( samples_per_pixel );

if  (mp11::mp_contains<color_space_t, alpha_t>::value)
{
std:: vector <uint16_t> extra_samples {EXTRASAMPLE_ASSOCALPHA};
this->_io_dev.template set_property<tiff_extra_samples>( extra_samples );
}

tiff_bits_per_sample::type bits_per_sample = detail::unsigned_integral_num_bits< channel_t >::value;
this->_io_dev.template set_property<tiff_bits_per_sample>( bits_per_sample );

tiff_sample_format::type sampl_format = detail::sample_format< channel_t >::value;
this->_io_dev.template set_property<tiff_sample_format>( sampl_format );

this->_io_dev.template set_property<tiff_photometric_interpretation>( this->_info._photometric_interpretation );

this->_io_dev.template set_property<tiff_compression>( this->_info._compression );

this->_io_dev.template set_property<tiff_orientation>( this->_info._orientation );

this->_io_dev.template set_property<tiff_rows_per_strip>( this->_io_dev.get_default_strip_size() );

this->_io_dev.template set_property<tiff_resolution_unit>( this->_info._resolution_unit );
this->_io_dev.template set_property<tiff_x_resolution>( this->_info._x_resolution );
this->_io_dev.template set_property<tiff_y_resolution>( this->_info._y_resolution );


if ( 0 != this->_info._icc_profile.size())
this->_io_dev.template set_property<tiff_icc_profile>( this->_info._icc_profile );
}


public:

Device _io_dev;

image_write_info< tiff_tag > _info;
};

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(pop)
#endif

} 
} 

#endif
