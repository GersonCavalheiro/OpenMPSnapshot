
#ifndef BOOST_INTERPROCESS_ANONYMOUS_SHARED_MEMORY_HPP
#define BOOST_INTERPROCESS_ANONYMOUS_SHARED_MEMORY_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/move/move.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <cstddef>

#if (!defined(BOOST_INTERPROCESS_WINDOWS))
#  include <fcntl.h>        
#  include <sys/mman.h>     
#  include <sys/stat.h>     
#else
#include <boost/interprocess/windows_shared_memory.hpp>
#endif



namespace boost {
namespace interprocess {

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

namespace ipcdetail{

class raw_mapped_region_creator
{
public:
static mapped_region
create_posix_mapped_region(void *address, std::size_t size)
{
mapped_region region;
region.m_base = address;
region.m_size = size;
return region;
}
};
}

#endif   

static mapped_region
anonymous_shared_memory(std::size_t size, void *address = 0)
#if (!defined(BOOST_INTERPROCESS_WINDOWS))
{
int flags;
int fd = -1;

#if defined(MAP_ANONYMOUS) 
flags = MAP_ANONYMOUS | MAP_SHARED;
#elif !defined(MAP_ANONYMOUS) && defined(MAP_ANON) 
flags = MAP_ANON | MAP_SHARED;
#else 
fd = open("/dev/zero", O_RDWR);
flags = MAP_SHARED;
if(fd == -1){
error_info err = system_error_code();
throw interprocess_exception(err);
}
#endif


address = mmap( address
, size
, PROT_READ|PROT_WRITE
, flags
, fd
, 0);

if(address == MAP_FAILED){
if(fd != -1)
close(fd);
error_info err = system_error_code();
throw interprocess_exception(err);
}

if(fd != -1)
close(fd);

return ipcdetail::raw_mapped_region_creator::create_posix_mapped_region(address, size);
}
#else
{
windows_shared_memory anonymous_mapping(create_only, 0, read_write, size);
return mapped_region(anonymous_mapping, read_write, 0, size, address);
}

#endif

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
