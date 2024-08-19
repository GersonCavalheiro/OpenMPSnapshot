
#ifndef BOOST_INTERPROCESS_CREATION_TAGS_HPP
#define BOOST_INTERPROCESS_CREATION_TAGS_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

namespace boost {
namespace interprocess {

struct create_only_t {};

struct open_only_t {};

struct open_read_only_t {};

struct open_read_private_t {};

struct open_copy_on_write_t {};

struct open_or_create_t {};

static const create_only_t    create_only    = create_only_t();

static const open_only_t      open_only      = open_only_t();

static const open_read_only_t open_read_only = open_read_only_t();

static const open_or_create_t open_or_create = open_or_create_t();

static const open_copy_on_write_t open_copy_on_write = open_copy_on_write_t();

namespace ipcdetail {

enum create_enum_t
{  DoCreate, DoOpen, DoOpenOrCreate   };

}  

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   

