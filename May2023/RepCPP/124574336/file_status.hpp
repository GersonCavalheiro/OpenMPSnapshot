




#ifndef BOOST_FILESYSTEM3_FILE_STATUS_HPP
#define BOOST_FILESYSTEM3_FILE_STATUS_HPP

#include <boost/config.hpp>

# if defined( BOOST_NO_STD_WSTRING )
#   error Configuration not supported: Boost.Filesystem V3 and later requires std::wstring support
# endif

#include <boost/filesystem/config.hpp>

#include <boost/detail/bitmask.hpp>

#include <boost/config/abi_prefix.hpp> 


namespace boost {
namespace filesystem {


enum file_type
{
status_error,
# ifndef BOOST_FILESYSTEM_NO_DEPRECATED
status_unknown = status_error,
# endif
file_not_found,
regular_file,
directory_file,
symlink_file,
block_file,
character_file,
fifo_file,
socket_file,
reparse_file,  
type_unknown,  

_detail_directory_symlink  
};


enum perms
{
no_perms = 0,       



owner_read = 0400,  
owner_write = 0200, 
owner_exe = 0100,   
owner_all = 0700,   

group_read = 040,   
group_write = 020,  
group_exe = 010,    
group_all = 070,    

others_read = 04,   
others_write = 02,  
others_exe = 01,    
others_all = 07,    

all_all = 0777,     


set_uid_on_exe = 04000, 
set_gid_on_exe = 02000, 
sticky_bit     = 01000, 

perms_mask = 07777,     

perms_not_known = 0xFFFF, 


add_perms = 0x1000,     
remove_perms = 0x2000,  

symlink_perms = 0x4000, 

_detail_extend_perms_32_1 = 0x7fffffff,
_detail_extend_perms_32_2 = -0x7fffffff-1
};

BOOST_BITMASK(perms)


class file_status
{
public:
BOOST_CONSTEXPR file_status() BOOST_NOEXCEPT :
m_value(status_error), m_perms(perms_not_known)
{
}
explicit BOOST_CONSTEXPR file_status(file_type v) BOOST_NOEXCEPT :
m_value(v), m_perms(perms_not_known)
{
}
BOOST_CONSTEXPR file_status(file_type v, perms prms) BOOST_NOEXCEPT :
m_value(v), m_perms(prms)
{
}


BOOST_CONSTEXPR file_status(const file_status& rhs) BOOST_NOEXCEPT :
m_value(rhs.m_value), m_perms(rhs.m_perms)
{
}
BOOST_CXX14_CONSTEXPR file_status& operator=(const file_status& rhs) BOOST_NOEXCEPT
{
m_value = rhs.m_value;
m_perms = rhs.m_perms;
return *this;
}

# if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
BOOST_CONSTEXPR file_status(file_status&& rhs) BOOST_NOEXCEPT :
m_value(static_cast< file_type&& >(rhs.m_value)), m_perms(static_cast< enum perms&& >(rhs.m_perms))
{
}
BOOST_CXX14_CONSTEXPR file_status& operator=(file_status&& rhs) BOOST_NOEXCEPT
{
m_value = static_cast< file_type&& >(rhs.m_value);
m_perms = static_cast< enum perms&& >(rhs.m_perms);
return *this;
}
# endif

BOOST_CONSTEXPR file_type  type() const BOOST_NOEXCEPT            { return m_value; }
BOOST_CONSTEXPR perms      permissions() const BOOST_NOEXCEPT     { return m_perms; }

BOOST_CXX14_CONSTEXPR void type(file_type v) BOOST_NOEXCEPT       { m_value = v; }
BOOST_CXX14_CONSTEXPR void permissions(perms prms) BOOST_NOEXCEPT { m_perms = prms; }

BOOST_CONSTEXPR bool operator==(const file_status& rhs) const BOOST_NOEXCEPT
{
return type() == rhs.type() && permissions() == rhs.permissions();
}
BOOST_CONSTEXPR bool operator!=(const file_status& rhs) const BOOST_NOEXCEPT
{
return !(*this == rhs);
}

private:
file_type   m_value;
enum perms  m_perms;
};

inline BOOST_CONSTEXPR bool type_present(file_status f) BOOST_NOEXCEPT
{
return f.type() != status_error;
}
inline BOOST_CONSTEXPR bool permissions_present(file_status f) BOOST_NOEXCEPT
{
return f.permissions() != perms_not_known;
}
inline BOOST_CONSTEXPR bool status_known(file_status f) BOOST_NOEXCEPT
{
return filesystem::type_present(f) && filesystem::permissions_present(f);
}
inline BOOST_CONSTEXPR bool exists(file_status f) BOOST_NOEXCEPT
{
return f.type() != status_error && f.type() != file_not_found;
}
inline BOOST_CONSTEXPR bool is_regular_file(file_status f) BOOST_NOEXCEPT
{
return f.type() == regular_file;
}
inline BOOST_CONSTEXPR bool is_directory(file_status f) BOOST_NOEXCEPT
{
return f.type() == directory_file;
}
inline BOOST_CONSTEXPR bool is_symlink(file_status f) BOOST_NOEXCEPT
{
return f.type() == symlink_file;
}
inline BOOST_CONSTEXPR bool is_other(file_status f) BOOST_NOEXCEPT
{
return filesystem::exists(f) && !filesystem::is_regular_file(f)
&& !filesystem::is_directory(f) && !filesystem::is_symlink(f);
}

# ifndef BOOST_FILESYSTEM_NO_DEPRECATED
inline bool is_regular(file_status f) BOOST_NOEXCEPT { return filesystem::is_regular_file(f); }
# endif

} 
} 

#include <boost/config/abi_suffix.hpp> 
#endif 
