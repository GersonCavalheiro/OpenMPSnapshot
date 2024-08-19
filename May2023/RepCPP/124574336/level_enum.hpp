#ifndef BOOST_SERIALIZATION_LEVEL_ENUM_HPP
#define BOOST_SERIALIZATION_LEVEL_ENUM_HPP

#if defined(_MSC_VER)
# pragma once
#endif




namespace boost {
namespace serialization {


enum level_type
{
not_serializable = 0,
primitive_type = 1,
object_serializable = 2,
object_class_info = 3
};

} 
} 

#endif 
