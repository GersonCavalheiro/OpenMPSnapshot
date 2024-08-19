



#ifndef BOOST_SYSTEM_LINUX_ERROR_HPP
#define BOOST_SYSTEM_LINUX_ERROR_HPP

#include <boost/config/pragma_message.hpp>

#if !defined(BOOST_ALLOW_DEPRECATED_HEADERS)
BOOST_PRAGMA_MESSAGE("This header is deprecated and is slated for removal."
" If you want it retained, please open an issue in github.com/boostorg/system.")
#endif


#if defined(linux) || defined(__linux) || defined(__linux__)

#include <boost/system/error_code.hpp>

namespace boost
{
namespace system
{


namespace linux_error
{
enum linux_errno
{
advertise_error = EADV,
bad_exchange = EBADE,
bad_file_number = EBADFD,
bad_font_format = EBFONT,
bad_request_code = EBADRQC,
bad_request_descriptor = EBADR,
bad_slot = EBADSLT,
channel_range = ECHRNG,
communication_error = ECOMM,
dot_dot_error = EDOTDOT,
exchange_full = EXFULL,
host_down = EHOSTDOWN,
is_named_file_type= EISNAM,
key_expired = EKEYEXPIRED,
key_rejected = EKEYREJECTED,
key_revoked = EKEYREVOKED,
level2_halt= EL2HLT,
level2_no_syncronized= EL2NSYNC,
level3_halt = EL3HLT,
level3_reset = EL3RST,
link_range = ELNRNG,
medium_type = EMEDIUMTYPE,
no_anode= ENOANO,
no_block_device = ENOTBLK,
no_csi = ENOCSI,
no_key = ENOKEY,
no_medium = ENOMEDIUM,
no_network = ENONET,
no_package = ENOPKG,
not_avail = ENAVAIL,
not_named_file_type= ENOTNAM,
not_recoverable = ENOTRECOVERABLE,
not_unique = ENOTUNIQ,
owner_dead = EOWNERDEAD,
protocol_no_supported = EPFNOSUPPORT,
remote_address_changed = EREMCHG,
remote_io_error = EREMOTEIO,
remote_object = EREMOTE,
restart_needed = ERESTART,
shared_library_access = ELIBACC,
shared_library_bad = ELIBBAD,
shared_library_execute = ELIBEXEC,
shared_library_max_ = ELIBMAX,
shared_library_section= ELIBSCN,
shutdown = ESHUTDOWN,
socket_type_not_supported = ESOCKTNOSUPPORT,
srmount_error = ESRMNT,
stream_pipe_error = ESTRPIPE,
too_many_references = ETOOMANYREFS,
too_many_users = EUSERS,
unattached = EUNATCH,
unclean = EUCLEAN
};
}  

# ifdef BOOST_SYSTEM_ENABLE_DEPRECATED
namespace Linux = linux_error;
# endif

template<> struct is_error_code_enum<linux_error::linux_errno>
{ static const bool value = true; };

namespace linux_error
{
inline error_code make_error_code( linux_errno e )
{ return error_code( e, system_category() ); }
}

}  
}  

#endif  

#endif  
