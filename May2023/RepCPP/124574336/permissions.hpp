
#ifndef BOOST_INTERPROCESS_PERMISSIONS_HPP
#define BOOST_INTERPROCESS_PERMISSIONS_HPP

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>

#if defined(BOOST_INTERPROCESS_WINDOWS)

#include <boost/interprocess/detail/win32_api.hpp>

#endif

#endif   


namespace boost {
namespace interprocess {

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

#if defined(BOOST_INTERPROCESS_WINDOWS)

namespace ipcdetail {

template <int Dummy>
struct unrestricted_permissions_holder
{
static winapi::interprocess_all_access_security unrestricted;
};

template<int Dummy>
winapi::interprocess_all_access_security unrestricted_permissions_holder<Dummy>::unrestricted;

}  

#endif   

#endif   

class permissions
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

#if defined(BOOST_INTERPROCESS_WINDOWS)
typedef void*  os_permissions_type;
#else
typedef int    os_permissions_type;
#endif
os_permissions_type  m_perm;

#endif   

public:
permissions(os_permissions_type type)
: m_perm(type)
{}

permissions()
{  set_default(); }

void set_default()
{
#if defined (BOOST_INTERPROCESS_WINDOWS)
m_perm = 0;
#else
m_perm = 0644;
#endif
}

void set_unrestricted()
{
#if defined (BOOST_INTERPROCESS_WINDOWS)
m_perm = &ipcdetail::unrestricted_permissions_holder<0>::unrestricted;
#else
m_perm = 0666;
#endif
}

void set_permissions(os_permissions_type perm)
{  m_perm = perm; }

os_permissions_type get_permissions() const
{  return m_perm; }
};

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   

