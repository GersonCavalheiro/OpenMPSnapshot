
#ifndef BOOST_ASIO_DETAIL_HANDLER_TRACKING_HPP
#define BOOST_ASIO_DETAIL_HANDLER_TRACKING_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

namespace boost {
namespace asio {

class execution_context;

} 
} 

#if defined(BOOST_ASIO_CUSTOM_HANDLER_TRACKING)
# include BOOST_ASIO_CUSTOM_HANDLER_TRACKING
#elif defined(BOOST_ASIO_ENABLE_HANDLER_TRACKING)
# include <boost/system/error_code.hpp>
# include <boost/asio/detail/cstdint.hpp>
# include <boost/asio/detail/static_mutex.hpp>
# include <boost/asio/detail/tss_ptr.hpp>
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

#if defined(BOOST_ASIO_CUSTOM_HANDLER_TRACKING)


# if !defined(BOOST_ASIO_ENABLE_HANDLER_TRACKING)
#  define BOOST_ASIO_ENABLE_HANDLER_TRACKING 1
# endif 

#elif defined(BOOST_ASIO_ENABLE_HANDLER_TRACKING)

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

BOOST_ASIO_DECL static void init();

class location
{
public:
BOOST_ASIO_DECL explicit location(const char* file,
int line, const char* func);

BOOST_ASIO_DECL ~location();

private:
location(const location&) BOOST_ASIO_DELETED;
location& operator=(const location&) BOOST_ASIO_DELETED;

friend class handler_tracking;
const char* file_;
int line_;
const char* func_;
location* next_;
};

BOOST_ASIO_DECL static void creation(
execution_context& context, tracked_handler& h,
const char* object_type, void* object,
uintmax_t native_handle, const char* op_name);

class completion
{
public:
BOOST_ASIO_DECL explicit completion(const tracked_handler& h);

BOOST_ASIO_DECL ~completion();

BOOST_ASIO_DECL void invocation_begin();

BOOST_ASIO_DECL void invocation_begin(const boost::system::error_code& ec);

BOOST_ASIO_DECL void invocation_begin(
const boost::system::error_code& ec, std::size_t bytes_transferred);

BOOST_ASIO_DECL void invocation_begin(
const boost::system::error_code& ec, int signal_number);

BOOST_ASIO_DECL void invocation_begin(
const boost::system::error_code& ec, const char* arg);

BOOST_ASIO_DECL void invocation_end();

private:
friend class handler_tracking;
uint64_t id_;
bool invoked_;
completion* next_;
};

BOOST_ASIO_DECL static void operation(execution_context& context,
const char* object_type, void* object,
uintmax_t native_handle, const char* op_name);

BOOST_ASIO_DECL static void reactor_registration(execution_context& context,
uintmax_t native_handle, uintmax_t registration);

BOOST_ASIO_DECL static void reactor_deregistration(execution_context& context,
uintmax_t native_handle, uintmax_t registration);

BOOST_ASIO_DECL static void reactor_events(execution_context& context,
uintmax_t registration, unsigned events);

BOOST_ASIO_DECL static void reactor_operation(
const tracked_handler& h, const char* op_name,
const boost::system::error_code& ec);

BOOST_ASIO_DECL static void reactor_operation(
const tracked_handler& h, const char* op_name,
const boost::system::error_code& ec, std::size_t bytes_transferred);

BOOST_ASIO_DECL static void write_line(const char* format, ...);

private:
struct tracking_state;
BOOST_ASIO_DECL static tracking_state* get_state();
};

# define BOOST_ASIO_INHERIT_TRACKED_HANDLER \
: public boost::asio::detail::handler_tracking::tracked_handler

# define BOOST_ASIO_ALSO_INHERIT_TRACKED_HANDLER \
, public boost::asio::detail::handler_tracking::tracked_handler

# define BOOST_ASIO_HANDLER_TRACKING_INIT \
boost::asio::detail::handler_tracking::init()

# define BOOST_ASIO_HANDLER_LOCATION(args) \
boost::asio::detail::handler_tracking::location tracked_location args

# define BOOST_ASIO_HANDLER_CREATION(args) \
boost::asio::detail::handler_tracking::creation args

# define BOOST_ASIO_HANDLER_COMPLETION(args) \
boost::asio::detail::handler_tracking::completion tracked_completion args

# define BOOST_ASIO_HANDLER_INVOCATION_BEGIN(args) \
tracked_completion.invocation_begin args

# define BOOST_ASIO_HANDLER_INVOCATION_END \
tracked_completion.invocation_end()

# define BOOST_ASIO_HANDLER_OPERATION(args) \
boost::asio::detail::handler_tracking::operation args

# define BOOST_ASIO_HANDLER_REACTOR_REGISTRATION(args) \
boost::asio::detail::handler_tracking::reactor_registration args

# define BOOST_ASIO_HANDLER_REACTOR_DEREGISTRATION(args) \
boost::asio::detail::handler_tracking::reactor_deregistration args

# define BOOST_ASIO_HANDLER_REACTOR_READ_EVENT 1
# define BOOST_ASIO_HANDLER_REACTOR_WRITE_EVENT 2
# define BOOST_ASIO_HANDLER_REACTOR_ERROR_EVENT 4

# define BOOST_ASIO_HANDLER_REACTOR_EVENTS(args) \
boost::asio::detail::handler_tracking::reactor_events args

# define BOOST_ASIO_HANDLER_REACTOR_OPERATION(args) \
boost::asio::detail::handler_tracking::reactor_operation args

#else 

# define BOOST_ASIO_INHERIT_TRACKED_HANDLER
# define BOOST_ASIO_ALSO_INHERIT_TRACKED_HANDLER
# define BOOST_ASIO_HANDLER_TRACKING_INIT (void)0
# define BOOST_ASIO_HANDLER_LOCATION(loc) (void)0
# define BOOST_ASIO_HANDLER_CREATION(args) (void)0
# define BOOST_ASIO_HANDLER_COMPLETION(args) (void)0
# define BOOST_ASIO_HANDLER_INVOCATION_BEGIN(args) (void)0
# define BOOST_ASIO_HANDLER_INVOCATION_END (void)0
# define BOOST_ASIO_HANDLER_OPERATION(args) (void)0
# define BOOST_ASIO_HANDLER_REACTOR_REGISTRATION(args) (void)0
# define BOOST_ASIO_HANDLER_REACTOR_DEREGISTRATION(args) (void)0
# define BOOST_ASIO_HANDLER_REACTOR_READ_EVENT 0
# define BOOST_ASIO_HANDLER_REACTOR_WRITE_EVENT 0
# define BOOST_ASIO_HANDLER_REACTOR_ERROR_EVENT 0
# define BOOST_ASIO_HANDLER_REACTOR_EVENTS(args) (void)0
# define BOOST_ASIO_HANDLER_REACTOR_OPERATION(args) (void)0

#endif 

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/detail/impl/handler_tracking.ipp>
#endif 

#endif 
