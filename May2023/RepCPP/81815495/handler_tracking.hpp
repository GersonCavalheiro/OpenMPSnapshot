
#ifndef ASIO_DETAIL_HANDLER_TRACKING_HPP
#define ASIO_DETAIL_HANDLER_TRACKING_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

namespace asio {

class execution_context;

} 

#if defined(ASIO_CUSTOM_HANDLER_TRACKING)
# include ASIO_CUSTOM_HANDLER_TRACKING
#elif defined(ASIO_ENABLE_HANDLER_TRACKING)
# include "asio/error_code.hpp"
# include "asio/detail/cstdint.hpp"
# include "asio/detail/static_mutex.hpp"
# include "asio/detail/tss_ptr.hpp"
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

#if defined(ASIO_CUSTOM_HANDLER_TRACKING)


# if !defined(ASIO_ENABLE_HANDLER_TRACKING)
#  define ASIO_ENABLE_HANDLER_TRACKING 1
# endif 

#elif defined(ASIO_ENABLE_HANDLER_TRACKING)

class handler_tracking
{
public:
class completion;

class tracked_handler
{
private:
friend class handler_tracking;
friend class completion;
uint64_t id_;

protected:
tracked_handler() : id_(0) {}

~tracked_handler() {}
};

ASIO_DECL static void init();

class location
{
public:
ASIO_DECL explicit location(const char* file,
int line, const char* func);

ASIO_DECL ~location();

private:
location(const location&) ASIO_DELETED;
location& operator=(const location&) ASIO_DELETED;

friend class handler_tracking;
const char* file_;
int line_;
const char* func_;
location* next_;
};

ASIO_DECL static void creation(
execution_context& context, tracked_handler& h,
const char* object_type, void* object,
uintmax_t native_handle, const char* op_name);

class completion
{
public:
ASIO_DECL explicit completion(const tracked_handler& h);

ASIO_DECL ~completion();

ASIO_DECL void invocation_begin();

ASIO_DECL void invocation_begin(const asio::error_code& ec);

ASIO_DECL void invocation_begin(
const asio::error_code& ec, std::size_t bytes_transferred);

ASIO_DECL void invocation_begin(
const asio::error_code& ec, int signal_number);

ASIO_DECL void invocation_begin(
const asio::error_code& ec, const char* arg);

ASIO_DECL void invocation_end();

private:
friend class handler_tracking;
uint64_t id_;
bool invoked_;
completion* next_;
};

ASIO_DECL static void operation(execution_context& context,
const char* object_type, void* object,
uintmax_t native_handle, const char* op_name);

ASIO_DECL static void reactor_registration(execution_context& context,
uintmax_t native_handle, uintmax_t registration);

ASIO_DECL static void reactor_deregistration(execution_context& context,
uintmax_t native_handle, uintmax_t registration);

ASIO_DECL static void reactor_events(execution_context& context,
uintmax_t registration, unsigned events);

ASIO_DECL static void reactor_operation(
const tracked_handler& h, const char* op_name,
const asio::error_code& ec);

ASIO_DECL static void reactor_operation(
const tracked_handler& h, const char* op_name,
const asio::error_code& ec, std::size_t bytes_transferred);

ASIO_DECL static void write_line(const char* format, ...);

private:
struct tracking_state;
ASIO_DECL static tracking_state* get_state();
};

# define ASIO_INHERIT_TRACKED_HANDLER \
: public asio::detail::handler_tracking::tracked_handler

# define ASIO_ALSO_INHERIT_TRACKED_HANDLER \
, public asio::detail::handler_tracking::tracked_handler

# define ASIO_HANDLER_TRACKING_INIT \
asio::detail::handler_tracking::init()

# define ASIO_HANDLER_LOCATION(args) \
asio::detail::handler_tracking::location tracked_location args

# define ASIO_HANDLER_CREATION(args) \
asio::detail::handler_tracking::creation args

# define ASIO_HANDLER_COMPLETION(args) \
asio::detail::handler_tracking::completion tracked_completion args

# define ASIO_HANDLER_INVOCATION_BEGIN(args) \
tracked_completion.invocation_begin args

# define ASIO_HANDLER_INVOCATION_END \
tracked_completion.invocation_end()

# define ASIO_HANDLER_OPERATION(args) \
asio::detail::handler_tracking::operation args

# define ASIO_HANDLER_REACTOR_REGISTRATION(args) \
asio::detail::handler_tracking::reactor_registration args

# define ASIO_HANDLER_REACTOR_DEREGISTRATION(args) \
asio::detail::handler_tracking::reactor_deregistration args

# define ASIO_HANDLER_REACTOR_READ_EVENT 1
# define ASIO_HANDLER_REACTOR_WRITE_EVENT 2
# define ASIO_HANDLER_REACTOR_ERROR_EVENT 4

# define ASIO_HANDLER_REACTOR_EVENTS(args) \
asio::detail::handler_tracking::reactor_events args

# define ASIO_HANDLER_REACTOR_OPERATION(args) \
asio::detail::handler_tracking::reactor_operation args

#else 

# define ASIO_INHERIT_TRACKED_HANDLER
# define ASIO_ALSO_INHERIT_TRACKED_HANDLER
# define ASIO_HANDLER_TRACKING_INIT (void)0
# define ASIO_HANDLER_LOCATION(loc) (void)0
# define ASIO_HANDLER_CREATION(args) (void)0
# define ASIO_HANDLER_COMPLETION(args) (void)0
# define ASIO_HANDLER_INVOCATION_BEGIN(args) (void)0
# define ASIO_HANDLER_INVOCATION_END (void)0
# define ASIO_HANDLER_OPERATION(args) (void)0
# define ASIO_HANDLER_REACTOR_REGISTRATION(args) (void)0
# define ASIO_HANDLER_REACTOR_DEREGISTRATION(args) (void)0
# define ASIO_HANDLER_REACTOR_READ_EVENT 0
# define ASIO_HANDLER_REACTOR_WRITE_EVENT 0
# define ASIO_HANDLER_REACTOR_ERROR_EVENT 0
# define ASIO_HANDLER_REACTOR_EVENTS(args) (void)0
# define ASIO_HANDLER_REACTOR_OPERATION(args) (void)0

#endif 

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/handler_tracking.ipp"
#endif 

#endif 
