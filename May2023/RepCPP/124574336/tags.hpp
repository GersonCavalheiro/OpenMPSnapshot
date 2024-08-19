#ifndef BOOST_GIL_EXTENSION_IO_RAW_TAGS_HPP
#define BOOST_GIL_EXTENSION_IO_RAW_TAGS_HPP

#include <boost/gil/io/base.hpp>

#ifdef _MSC_VER
#ifndef WIN32
#define WIN32
#endif
#pragma warning(push)
#pragma warning(disable:4251) 
#endif

#include <libraw/libraw.h>

namespace boost { namespace gil {

struct raw_tag : format_tag {};

struct raw_image_width : property_base< int32_t > {};

struct raw_image_height : property_base< int32_t > {};

struct raw_samples_per_pixel : property_base< int32_t > {};

struct raw_bits_per_pixel : property_base< int32_t > {};

struct raw_camera_manufacturer : property_base< std::string > {};

struct raw_camera_model : property_base< std::string > {};

struct raw_raw_images_count : property_base< unsigned > {};

struct raw_dng_version : property_base< unsigned > {};

struct raw_number_colors : property_base< int32_t > {};

struct raw_colors_description : property_base< std::string > {};

struct raw_raw_height : property_base< uint16_t > {};

struct raw_raw_width : property_base< uint16_t > {};

struct raw_visible_height : property_base< uint16_t > {};

struct raw_visible_width : property_base< uint16_t > {};

struct raw_top_margin : property_base< uint16_t > {};

struct raw_left_margin : property_base< uint16_t > {};

struct raw_output_height : property_base< uint16_t > {};

struct raw_output_width : property_base< uint16_t > {};

struct raw_pixel_aspect : property_base< double > {};

struct raw_flip : property_base< uint32_t > {};

struct raw_iso_speed : property_base< float > {};

struct raw_shutter : property_base< float > {};

struct raw_aperture : property_base< float > {};

struct raw_focal_length : property_base< float > {};

struct raw_timestamp : property_base< time_t > {};

struct raw_shot_order : property_base< uint16_t > {};

struct raw_image_description : property_base< std::string > {};

struct raw_artist : property_base< std::string > {};

struct raw_libraw_version : property_base< std::string > {};

struct raw_unpack_function_name : property_base< std::string > {};

template<>
struct image_read_info< raw_tag >
{
image_read_info< raw_tag >()
: _valid( false )
{}

raw_image_width::type       _width;
raw_image_height::type      _height;
raw_samples_per_pixel::type _samples_per_pixel;
raw_bits_per_pixel::type    _bits_per_pixel;

raw_camera_manufacturer::type _camera_manufacturer;
raw_camera_model::type        _camera_model;

raw_raw_images_count::type   _raw_images_count;
raw_dng_version::type        _dng_version;
raw_number_colors::type      _number_colors;
raw_colors_description::type _colors_description;

raw_raw_width::type      _raw_width;
raw_raw_height::type     _raw_height;
raw_visible_width::type  _visible_width;
raw_visible_height::type _visible_height;
raw_top_margin::type     _top_margin;
raw_left_margin::type    _left_margin;
raw_output_width::type   _output_width;
raw_output_height::type  _output_height;
raw_pixel_aspect::type   _pixel_aspect;
raw_flip::type           _flip;

raw_iso_speed::type         _iso_speed;
raw_shutter::type           _shutter;
raw_aperture::type          _aperture;
raw_focal_length::type      _focal_length;
raw_timestamp::type         _timestamp;
raw_shot_order::type        _shot_order;
raw_image_description::type _image_description;
raw_artist::type            _artist;

raw_libraw_version::type       _libraw_version;
raw_unpack_function_name::type _unpack_function_name;

bool _valid;
};

template<>
struct image_read_settings< raw_tag > : public image_read_settings_base
{
image_read_settings()
: image_read_settings_base()
{}

image_read_settings( const point_t& top_left
, const point_t& dim
)
: image_read_settings_base( top_left
, dim
)
{}
};

template<>
struct image_write_info< raw_tag >
{
};

}} 

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif
