


#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>
#include <hydra/detail/external/hydra_thrust/detail/modern_gcc_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011 && !defined(HYDRA_THRUST_LEGACY_GCC)

#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/system/error_code.h>

#include <stdexcept>

HYDRA_THRUST_BEGIN_NS

enum class event_errc
{
unknown_event_error
, no_state
, no_content
, last_event_error
};

inline error_code make_error_code(event_errc e);

inline error_condition make_error_condition(event_errc e);

struct event_error_category : error_category
{
event_error_category() = default;

virtual char const* name() const
{
return "event";
}

virtual std::string message(int ev) const
{
switch (static_cast<event_errc>(ev))
{
case event_errc::no_state:
{
return "no_state: an operation that requires an event or future to have "
"a stream or content has been performed on a event or future "
"without either, e.g. a moved-from or default constructed event "
"or future (anevent or future may have been consumed more than "
"once)";
}
case event_errc::no_content:
{
return "no_content: an operation that requires a future to have content "
"has been performed on future without any, e.g. a moved-from, "
"default constructed, or `hydra_thrust::new_stream` constructed future "
"(a future may have been consumed more than once)";
}
default:
{
return "unknown_event_error: an unknown error with a future "
"object has occurred";
}
};
}

virtual error_condition default_error_condition(int ev) const
{
if (
event_errc::last_event_error
>
static_cast<event_errc>(ev)
)
return make_error_condition(static_cast<event_errc>(ev));

return system_category().default_error_condition(ev);
}
}; 

inline error_category const& event_category()
{
static const event_error_category result;
return result;
}

template<> struct is_error_code_enum<event_errc> : true_type {};

inline error_code make_error_code(event_errc e)
{
return error_code(static_cast<int>(e), event_category());
}

inline error_condition make_error_condition(event_errc e)
{
return error_condition(static_cast<int>(e), event_category());
} 

struct event_error : std::logic_error
{
__host__
explicit event_error(error_code ec)
: std::logic_error(ec.message()), ec_(ec)
{}

__host__
explicit event_error(event_errc e)
: event_error(make_error_code(e))
{}

__host__
error_code const& code() const noexcept
{
return ec_;
}

__host__
virtual ~event_error() noexcept {}

private:
error_code ec_;
};

inline bool operator==(event_error const& lhs, event_error const& rhs) noexcept
{
return lhs.code() == rhs.code();
}

inline bool operator<(event_error const& lhs, event_error const& rhs) noexcept
{
return lhs.code() < rhs.code();
}

HYDRA_THRUST_END_NS

#endif

