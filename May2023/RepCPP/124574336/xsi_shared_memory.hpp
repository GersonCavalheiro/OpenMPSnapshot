
#ifndef BOOST_INTERPROCESS_XSI_SHARED_MEMORY_HPP
#define BOOST_INTERPROCESS_XSI_SHARED_MEMORY_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#if !defined(BOOST_INTERPROCESS_XSI_SHARED_MEMORY_OBJECTS)
#error "This header can't be used in operating systems without XSI (System V) shared memory support"
#endif

#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/detail/utilities.hpp>

#include <boost/interprocess/detail/os_file_functions.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/xsi_key.hpp>
#include <boost/interprocess/permissions.hpp>
#include <boost/interprocess/detail/simple_swap.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/cstdint.hpp>
#include <cstddef>
#include <sys/shm.h>



namespace boost {
namespace interprocess {

class xsi_shared_memory
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
BOOST_MOVABLE_BUT_NOT_COPYABLE(xsi_shared_memory)
#endif   

public:
xsi_shared_memory();

xsi_shared_memory(open_only_t, int shmid)
: m_shmid (shmid)
{}

xsi_shared_memory(create_only_t, const xsi_key &key, std::size_t size, const permissions& perm = permissions())
{  this->priv_open_or_create(ipcdetail::DoCreate, key, perm, size);  }

xsi_shared_memory(open_or_create_t, const xsi_key &key, std::size_t size, const permissions& perm = permissions())
{  this->priv_open_or_create(ipcdetail::DoOpenOrCreate, key, perm, size);  }

xsi_shared_memory(open_only_t, const xsi_key &key)
{  this->priv_open_or_create(ipcdetail::DoOpen, key, permissions(), 0);  }

xsi_shared_memory(BOOST_RV_REF(xsi_shared_memory) moved)
: m_shmid(-1)
{  this->swap(moved);   }

xsi_shared_memory &operator=(BOOST_RV_REF(xsi_shared_memory) moved)
{
xsi_shared_memory tmp(boost::move(moved));
this->swap(tmp);
return *this;
}

void swap(xsi_shared_memory &other);

~xsi_shared_memory();

int get_shmid() const;

mapping_handle_t get_mapping_handle() const;

static bool remove(int shmid);

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:

bool priv_open_or_create( ipcdetail::create_enum_t type
, const xsi_key &key
, const permissions& perm
, std::size_t size);
int            m_shmid;
#endif   
};

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

inline xsi_shared_memory::xsi_shared_memory()
:  m_shmid(-1)
{}

inline xsi_shared_memory::~xsi_shared_memory()
{}

inline int xsi_shared_memory::get_shmid() const
{  return m_shmid; }

inline void xsi_shared_memory::swap(xsi_shared_memory &other)
{
(simple_swap)(m_shmid, other.m_shmid);
}

inline mapping_handle_t xsi_shared_memory::get_mapping_handle() const
{  mapping_handle_t mhnd = { m_shmid, true};   return mhnd;   }

inline bool xsi_shared_memory::priv_open_or_create
(ipcdetail::create_enum_t type, const xsi_key &key, const permissions& permissions, std::size_t size)
{
int perm = permissions.get_permissions();
perm &= 0x01FF;
int shmflg = perm;

switch(type){
case ipcdetail::DoOpen:
shmflg |= 0;
break;
case ipcdetail::DoCreate:
shmflg |= IPC_CREAT | IPC_EXCL;
break;
case ipcdetail::DoOpenOrCreate:
shmflg |= IPC_CREAT;
break;
default:
{
error_info err = other_error;
throw interprocess_exception(err);
}
}

int ret = ::shmget(key.get_key(), size, shmflg);
int shmid = ret;
if((type == ipcdetail::DoOpen) && (-1 != ret)){
::shmid_ds xsi_ds;
ret = ::shmctl(ret, IPC_STAT, &xsi_ds);
size = xsi_ds.shm_segsz;
}
if(-1 == ret){
error_info err = system_error_code();
throw interprocess_exception(err);
}

m_shmid = shmid;
return true;
}

inline bool xsi_shared_memory::remove(int shmid)
{  return -1 != ::shmctl(shmid, IPC_RMID, 0); }

#endif   

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
