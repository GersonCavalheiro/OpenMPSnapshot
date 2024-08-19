




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/system/detail/errno.h>
#include <iostream>

namespace hydra_thrust
{

namespace system
{




class error_condition;
class error_code;


template<typename T> struct is_error_code_enum : public hydra_thrust::detail::false_type {};


template<typename T> struct is_error_condition_enum : public hydra_thrust::detail::false_type {};


namespace errc
{


enum errc_t
{
address_family_not_supported       = detail::eafnosupport,
address_in_use                     = detail::eaddrinuse,
address_not_available              = detail::eaddrnotavail,
already_connected                  = detail::eisconn,
argument_list_too_long             = detail::e2big,
argument_out_of_domain             = detail::edom,
bad_address                        = detail::efault,
bad_file_descriptor                = detail::ebadf,
bad_message                        = detail::ebadmsg,
broken_pipe                        = detail::epipe,
connection_aborted                 = detail::econnaborted,
connection_already_in_progress     = detail::ealready,
connection_refused                 = detail::econnrefused,
connection_reset                   = detail::econnreset,
cross_device_link                  = detail::exdev,
destination_address_required       = detail::edestaddrreq,
device_or_resource_busy            = detail::ebusy,
directory_not_empty                = detail::enotempty,
executable_format_error            = detail::enoexec,
file_exists                        = detail::eexist,
file_too_large                     = detail::efbig,
filename_too_long                  = detail::enametoolong,
function_not_supported             = detail::enosys,
host_unreachable                   = detail::ehostunreach,
identifier_removed                 = detail::eidrm,
illegal_byte_sequence              = detail::eilseq,
inappropriate_io_control_operation = detail::enotty,
interrupted                        = detail::eintr,
invalid_argument                   = detail::einval,
invalid_seek                       = detail::espipe,
io_error                           = detail::eio,
is_a_directory                     = detail::eisdir,
message_size                       = detail::emsgsize,
network_down                       = detail::enetdown,
network_reset                      = detail::enetreset,
network_unreachable                = detail::enetunreach,
no_buffer_space                    = detail::enobufs,
no_child_process                   = detail::echild,
no_link                            = detail::enolink,
no_lock_available                  = detail::enolck,
no_message_available               = detail::enodata,
no_message                         = detail::enomsg,
no_protocol_option                 = detail::enoprotoopt,
no_space_on_device                 = detail::enospc,
no_stream_resources                = detail::enosr,
no_such_device_or_address          = detail::enxio,
no_such_device                     = detail::enodev,
no_such_file_or_directory          = detail::enoent,
no_such_process                    = detail::esrch,
not_a_directory                    = detail::enotdir,
not_a_socket                       = detail::enotsock,
not_a_stream                       = detail::enostr,
not_connected                      = detail::enotconn,
not_enough_memory                  = detail::enomem,
not_supported                      = detail::enotsup,
operation_canceled                 = detail::ecanceled,
operation_in_progress              = detail::einprogress,
operation_not_permitted            = detail::eperm,
operation_not_supported            = detail::eopnotsupp,
operation_would_block              = detail::ewouldblock,
owner_dead                         = detail::eownerdead,
permission_denied                  = detail::eacces,
protocol_error                     = detail::eproto,
protocol_not_supported             = detail::eprotonosupport,
read_only_file_system              = detail::erofs,
resource_deadlock_would_occur      = detail::edeadlk,
resource_unavailable_try_again     = detail::eagain,
result_out_of_range                = detail::erange,
state_not_recoverable              = detail::enotrecoverable,
stream_timeout                     = detail::etime,
text_file_busy                     = detail::etxtbsy,
timed_out                          = detail::etimedout,
too_many_files_open_in_system      = detail::enfile,
too_many_files_open                = detail::emfile,
too_many_links                     = detail::emlink,
too_many_symbolic_link_levels      = detail::eloop,
value_too_large                    = detail::eoverflow,
wrong_protocol_type                = detail::eprototype
}; 

} 



template<> struct is_error_condition_enum<errc::errc_t> : public hydra_thrust::detail::true_type {};




class error_category
{
public:

inline virtual ~error_category(void);



inline virtual const char *name(void) const = 0;


inline virtual error_condition default_error_condition(int ev) const;


inline virtual bool equivalent(int code, const error_condition &condition) const;


inline virtual bool equivalent(const error_code &code, int condition) const;


virtual std::string message(int ev) const = 0;


inline bool operator==(const error_category &rhs) const;


inline bool operator!=(const error_category &rhs) const;


inline bool operator<(const error_category &rhs) const;
}; 





inline const error_category &generic_category(void);



inline const error_category &system_category(void);





class error_code
{
public:


inline error_code(void);


inline error_code(int val, const error_category &cat);


template <typename ErrorCodeEnum>
error_code(ErrorCodeEnum e
#if HYDRA_THRUST_HOST_COMPILER != HYDRA_THRUST_HOST_COMPILER_MSVC
, typename hydra_thrust::detail::enable_if<is_error_code_enum<ErrorCodeEnum>::value>::type * = 0
#endif 
);



inline void assign(int val, const error_category &cat);


template <typename ErrorCodeEnum>
#if HYDRA_THRUST_HOST_COMPILER != HYDRA_THRUST_HOST_COMPILER_MSVC
typename hydra_thrust::detail::enable_if<is_error_code_enum<ErrorCodeEnum>::value, error_code>::type &
#else
error_code &
#endif 
operator=(ErrorCodeEnum e);


inline void clear(void);



inline int value(void) const;


inline const error_category &category(void) const;


inline error_condition default_error_condition(void) const;


inline std::string message(void) const;



inline operator bool (void) const;


private:
int m_val;
const error_category *m_cat;

}; 





inline error_code make_error_code(errc::errc_t e);



inline bool operator<(const error_code &lhs, const error_code &rhs);



template <typename charT, typename traits>
std::basic_ostream<charT,traits>&
operator<<(std::basic_ostream<charT,traits>& os, const error_code &ec);





class error_condition
{
public:


inline error_condition(void);


inline error_condition(int val, const error_category &cat);


template<typename ErrorConditionEnum>
error_condition(ErrorConditionEnum e
#if HYDRA_THRUST_HOST_COMPILER != HYDRA_THRUST_HOST_COMPILER_MSVC
, typename hydra_thrust::detail::enable_if<is_error_condition_enum<ErrorConditionEnum>::value>::type * = 0
#endif 
);



inline void assign(int val, const error_category &cat);


template<typename ErrorConditionEnum>
#if HYDRA_THRUST_HOST_COMPILER != HYDRA_THRUST_HOST_COMPILER_MSVC
typename hydra_thrust::detail::enable_if<is_error_condition_enum<ErrorConditionEnum>::value, error_condition>::type &
#else
error_condition &
#endif 
operator=(ErrorConditionEnum e);


inline void clear(void);



inline int value(void) const;


inline const error_category &category(void) const;


inline std::string message(void) const;



inline operator bool (void) const;



private:
int m_val;
const error_category *m_cat;


}; 





inline error_condition make_error_condition(errc::errc_t e);



inline bool operator<(const error_condition &lhs, const error_condition &rhs);





inline bool operator==(const error_code &lhs, const error_code &rhs);



inline bool operator==(const error_code &lhs, const error_condition &rhs);



inline bool operator==(const error_condition &lhs, const error_code &rhs);



inline bool operator==(const error_condition &lhs, const error_condition &rhs);



inline bool operator!=(const error_code &lhs, const error_code &rhs);



inline bool operator!=(const error_code &lhs, const error_condition &rhs);



inline bool operator!=(const error_condition &lhs, const error_code &rhs);



inline bool operator!=(const error_condition &lhs, const error_condition &rhs);




} 


using system::error_category;
using system::error_code;
using system::error_condition;
using system::is_error_code_enum;
using system::is_error_condition_enum;
using system::make_error_code;
using system::make_error_condition;

namespace errc = system::errc;

using system::generic_category;
using system::system_category;

} 

#include <hydra/detail/external/hydra_thrust/system/detail/error_category.inl>
#include <hydra/detail/external/hydra_thrust/system/detail/error_code.inl>
#include <hydra/detail/external/hydra_thrust/system/detail/error_condition.inl>

