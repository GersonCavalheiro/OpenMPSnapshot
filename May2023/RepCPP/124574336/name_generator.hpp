


#ifndef BOOST_UUID_NAME_GENERATOR_HPP
#define BOOST_UUID_NAME_GENERATOR_HPP

#include <boost/config.hpp>
#include <boost/uuid/name_generator_sha1.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace uuids {

typedef name_generator_sha1 name_generator;

typedef name_generator_sha1 name_generator_latest;

} 
} 

#endif 
