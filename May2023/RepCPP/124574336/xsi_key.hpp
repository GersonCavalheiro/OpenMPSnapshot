
#ifndef BOOST_INTERPROCESS_XSI_KEY_HPP
#define BOOST_INTERPROCESS_XSI_KEY_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/detail/workaround.hpp>

#if !defined(BOOST_INTERPROCESS_XSI_SHARED_MEMORY_OBJECTS)
#error "This header can't be used in operating systems without XSI (System V) shared memory support"
#endif

#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <sys/types.h>
#include <sys/ipc.h>
#include <cstddef>
#include <boost/cstdint.hpp>


namespace boost {
namespace interprocess {

class xsi_key
{
public:

xsi_key()
: m_key(IPC_PRIVATE)
{}

explicit xsi_key(key_t key)
: m_key(key)
{}

xsi_key(const char *path, boost::uint8_t id)
{
key_t key;
if(path){
key  = ::ftok(path, id);
if(((key_t)-1) == key){
error_info err = system_error_code();
throw interprocess_exception(err);
}
}
else{
key = IPC_PRIVATE;
}
m_key = key;
}

key_t get_key() const
{  return m_key;  }

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
key_t m_key;
#endif   
};

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
