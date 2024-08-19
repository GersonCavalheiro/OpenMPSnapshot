

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>
#include <hydra/detail/external/hydra_thrust/detail/modern_gcc_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011 && !defined(HYDRA_THRUST_LEGACY_GCC)

#include <hydra/detail/external/hydra_thrust/optional.h>
#include <hydra/detail/external/hydra_thrust/detail/type_deduction.h>
#include <hydra/detail/external/hydra_thrust/type_traits/integer_sequence.h>
#include <hydra/detail/external/hydra_thrust/type_traits/remove_cvref.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/pointer_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/tuple_algorithms.h>
#include <hydra/detail/external/hydra_thrust/allocate_unique.h>
#include <hydra/detail/external/hydra_thrust/detail/static_assert.h>
#include <hydra/detail/external/hydra_thrust/detail/execute_with_dependencies.h>
#include <hydra/detail/external/hydra_thrust/detail/event_error.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/memory.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/future.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/util.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/get_value.h>

#include <type_traits>
#include <memory>

HYDRA_THRUST_BEGIN_NS

struct new_stream_t;

namespace system { namespace cuda { namespace detail
{


struct nonowning_t final {};

HYDRA_THRUST_INLINE_CONSTANT nonowning_t nonowning{};


struct marker_deleter final
{
__host__
void operator()(CUevent_st* e) const
{
if (nullptr != e)
hydra_thrust::cuda_cub::throw_on_error(cudaEventDestroy(e));
}
};


struct unique_marker final
{
using native_handle_type = CUevent_st*;

private:
std::unique_ptr<CUevent_st, marker_deleter> handle_;

public:
__host__
unique_marker()
: handle_(nullptr, marker_deleter())
{
native_handle_type e;
hydra_thrust::cuda_cub::throw_on_error(
cudaEventCreateWithFlags(&e, cudaEventDisableTiming)
);
handle_.reset(e);
}

__hydra_thrust_exec_check_disable__
unique_marker(unique_marker const&) = delete;
__hydra_thrust_exec_check_disable__
unique_marker(unique_marker&&) = default;
__hydra_thrust_exec_check_disable__
unique_marker& operator=(unique_marker const&) = delete;
__hydra_thrust_exec_check_disable__
unique_marker& operator=(unique_marker&&) = default;

__hydra_thrust_exec_check_disable__
~unique_marker() = default;

__host__
auto get() const
HYDRA_THRUST_DECLTYPE_RETURNS(native_handle_type(handle_.get()));
__host__
auto native_handle() const
HYDRA_THRUST_DECLTYPE_RETURNS(native_handle_type(handle_.get()));

__host__
bool valid() const noexcept { return bool(handle_); }

__host__
bool ready() const
{
cudaError_t const err = cudaEventQuery(handle_.get());

if (cudaErrorNotReady == err)
return false;

hydra_thrust::cuda_cub::throw_on_error(err);

return true;
}

__host__
void wait() const
{
hydra_thrust::cuda_cub::throw_on_error(cudaEventSynchronize(handle_.get()));
}

__host__
bool operator==(unique_marker const& other) const
{
return other.handle_ == handle_;
}

__host__
bool operator!=(unique_marker const& other) const
{
return !(other == *this);
}
};


struct stream_deleter final
{
__host__
void operator()(CUstream_st* s) const
{
if (nullptr != s)
hydra_thrust::cuda_cub::throw_on_error(cudaStreamDestroy(s));
}
};

struct stream_conditional_deleter final
{
private:
bool const cond_;

public:
__host__
constexpr stream_conditional_deleter() noexcept
: cond_(true) {}

__host__
explicit constexpr stream_conditional_deleter(nonowning_t) noexcept
: cond_(false) {}

__host__
void operator()(CUstream_st* s) const
{
if (cond_ && nullptr != s)
{
hydra_thrust::cuda_cub::throw_on_error(cudaStreamDestroy(s));
}
}
};


struct unique_stream final
{
using native_handle_type = CUstream_st*;

private:
std::unique_ptr<CUstream_st, stream_conditional_deleter> handle_;

public:
__host__
unique_stream()
: handle_(nullptr, stream_conditional_deleter())
{
native_handle_type s;
hydra_thrust::cuda_cub::throw_on_error(
cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)
);
handle_.reset(s);
}

__host__
explicit unique_stream(nonowning_t, native_handle_type handle)
: handle_(handle, stream_conditional_deleter(nonowning))
{}

__hydra_thrust_exec_check_disable__
unique_stream(unique_stream const&) = delete;
__hydra_thrust_exec_check_disable__
unique_stream(unique_stream&&) = default;
__hydra_thrust_exec_check_disable__
unique_stream& operator=(unique_stream const&) = delete;
__hydra_thrust_exec_check_disable__
unique_stream& operator=(unique_stream&&) = default;

__hydra_thrust_exec_check_disable__
~unique_stream() = default;

__host__
auto get() const
HYDRA_THRUST_DECLTYPE_RETURNS(native_handle_type(handle_.get()));
__host__
auto native_handle() const
HYDRA_THRUST_DECLTYPE_RETURNS(native_handle_type(handle_.get()));

__host__
bool valid() const noexcept { return bool(handle_); }

__host__
bool ready() const
{
cudaError_t const err = cudaStreamQuery(handle_.get());

if (cudaErrorNotReady == err)
return false;

hydra_thrust::cuda_cub::throw_on_error(err);

return true;
}

__host__
void wait() const
{
hydra_thrust::cuda_cub::throw_on_error(
cudaStreamSynchronize(handle_.get())
);
}

__host__
void depend_on(unique_marker& e)
{
hydra_thrust::cuda_cub::throw_on_error(
cudaStreamWaitEvent(handle_.get(), e.get(), 0)
);
}

__host__
void depend_on(unique_stream& s)
{
if (s != *this)
{
unique_marker e;
s.record(e);
depend_on(e);
}
}

__host__
void record(unique_marker& e)
{
hydra_thrust::cuda_cub::throw_on_error(cudaEventRecord(e.get(), handle_.get()));
}

__host__
bool operator==(unique_stream const& other) const
{
return other.handle_ == handle_;
}

__host__
bool operator!=(unique_stream const& other) const
{
return !(other == *this);
}
};



struct async_signal;

template <typename KeepAlives>
struct async_keep_alives ;

template <typename T>
struct async_value ;

template <typename T, typename Pointer, typename KeepAlives>
struct async_addressable_value_with_keep_alives
;


template <typename T, typename Pointer>
struct weak_promise;

template <typename X, typename XPointer = pointer<X>>
struct unique_eager_future_promise_pair final
{
unique_eager_future<X>    future;
weak_promise<X, XPointer> promise;
};

struct acquired_stream final
{
unique_stream stream;
optional<std::size_t> const acquired_from;
};

template <typename X, typename Y, typename Deleter>
__host__
optional<unique_stream>
try_acquire_stream(int device, std::unique_ptr<Y, Deleter>&) noexcept;

inline __host__
optional<unique_stream>
try_acquire_stream(int, unique_stream& stream) noexcept;

inline __host__
optional<unique_stream>
try_acquire_stream(int device, ready_event&) noexcept;

template <typename X>
inline __host__
optional<unique_stream>
try_acquire_stream(int device, ready_future<X>&) noexcept;

inline __host__
optional<unique_stream>
try_acquire_stream(int device, unique_eager_event& parent) noexcept;

template <typename X>
__host__
optional<unique_stream>
try_acquire_stream(int device, unique_eager_future<X>& parent) noexcept;

template <typename... Dependencies>
__host__
acquired_stream acquire_stream(int device, Dependencies&... deps) noexcept;

template <typename... Dependencies>
__host__
unique_eager_event
make_dependent_event(
std::tuple<Dependencies...>&& deps
);

template <
typename X, typename XPointer
, typename ComputeContent, typename... Dependencies
>
__host__
unique_eager_future_promise_pair<X, XPointer>
make_dependent_future(ComputeContent&& cc, std::tuple<Dependencies...>&& deps);


struct async_signal
{
protected:
unique_stream stream_;

public:
__host__
explicit async_signal(unique_stream&& stream)
: stream_(std::move(stream))
{}

__host__
virtual ~async_signal() {}

unique_stream&       stream()       noexcept { return stream_; }
unique_stream const& stream() const noexcept { return stream_; }
};

template <typename... KeepAlives>
struct async_keep_alives<std::tuple<KeepAlives...>> : virtual async_signal
{
using keep_alives_type = std::tuple<KeepAlives...>;

protected:
keep_alives_type keep_alives_;

public:
__host__
explicit async_keep_alives(
unique_stream&& stream, keep_alives_type&& keep_alives
)
: async_signal(std::move(stream))
, keep_alives_(std::move(keep_alives))
{}

__host__
virtual ~async_keep_alives() {}
};

template <typename T>
struct async_value : virtual async_signal
{
using value_type        = T;
using raw_const_pointer = value_type const*;

__host__
explicit async_value(unique_stream stream)
: async_signal(std::move(stream))
{}

__host__
virtual ~async_value() {}

__host__
virtual bool valid_content() const noexcept { return false; }

__host__
virtual value_type get()
{
throw hydra_thrust::event_error(event_errc::no_state);
}

__host__
virtual value_type extract()
{
throw hydra_thrust::event_error(event_errc::no_state);
}

#if defined(HYDRA_THRUST_ENABLE_FUTURE_RAW_DATA_MEMBER)
__host__
virtual raw_const_pointer raw_data() const
{
return nullptr;
}
#endif
};

template <typename T, typename Pointer, typename... KeepAlives>
struct async_addressable_value_with_keep_alives<
T, Pointer, std::tuple<KeepAlives...>
> final
: async_value<T>, async_keep_alives<std::tuple<KeepAlives...>>
{
using value_type        = typename async_value<T>::value_type;
using raw_const_pointer = typename async_value<T>::raw_const_pointer;

using keep_alives_type
= typename async_keep_alives<std::tuple<KeepAlives...>>::keep_alives_type;

using pointer
= typename hydra_thrust::detail::pointer_traits<Pointer>::template
rebind<value_type>::other;
using const_pointer
= typename hydra_thrust::detail::pointer_traits<Pointer>::template
rebind<value_type const>::other;

private:
pointer content_;

public:
HYDRA_THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING_BEGIN
template <typename ComputeContent>
__host__
explicit async_addressable_value_with_keep_alives(
unique_stream&&    stream
, keep_alives_type&& keep_alives
, ComputeContent&&   compute_content
)
: async_signal(std::move(stream))
, async_value<T>(std::move(stream))
, async_keep_alives<keep_alives_type>(
std::move(stream), std::move(keep_alives)
)
{
content_ = HYDRA_THRUST_FWD(compute_content)(std::get<0>(this->keep_alives_));
}
HYDRA_THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING_END

__host__
bool valid_content() const noexcept final override
{
return nullptr != content_;
}

__host__
pointer data() 
{
if (!valid_content())
throw hydra_thrust::event_error(event_errc::no_content);

return content_;
}

__host__
const_pointer data() const 
{
if (!valid_content())
throw hydra_thrust::event_error(event_errc::no_content);

return content_;
}

__host__
value_type get() final override
{
this->stream().wait();
return *data();
}

__host__
value_type extract() final override
{
this->stream().wait();
return std::move(*data());
}

#if defined(HYDRA_THRUST_ENABLE_FUTURE_RAW_DATA_MEMBER)
__host__
raw_const_pointer raw_data() const final override
{
return raw_pointer_cast(content_);
}
#endif
};


template <typename T, typename Pointer>
struct weak_promise final
{
using value_type = typename async_value<T>::value_type;

using pointer
= typename hydra_thrust::detail::pointer_traits<Pointer>::template
rebind<T>::other;
using const_pointer
= typename hydra_thrust::detail::pointer_traits<Pointer>::template
rebind<T const>::other;

private:
int device_ = 0;
pointer content_;

explicit weak_promise(int device, pointer content)
: device_(device), content_(std::move(content))
{}

public:
__host__ __device__
weak_promise() : device_(0), content_{} {}

__hydra_thrust_exec_check_disable__
weak_promise(weak_promise const&) = default;
__hydra_thrust_exec_check_disable__
weak_promise(weak_promise&&) = default;
__hydra_thrust_exec_check_disable__
weak_promise& operator=(weak_promise const&) = default;
__hydra_thrust_exec_check_disable__
weak_promise& operator=(weak_promise&&) = default;

template <typename U>
__host__ __device__
void set_value(U&& value) &&
{
*content_ = HYDRA_THRUST_FWD(value);
}

template <
typename X, typename XPointer
, typename ComputeContent, typename... Dependencies
>
friend __host__
unique_eager_future_promise_pair<X, XPointer>
hydra_thrust::system::cuda::detail::make_dependent_future(
ComputeContent&& cc, std::tuple<Dependencies...>&& deps
);
};


} 

struct ready_event final
{
ready_event() = default;

template <typename U>
__host__ __device__
explicit ready_event(ready_future<U>) {}

__host__ __device__
static constexpr bool valid_content() noexcept { return true; }

__host__ __device__
static constexpr bool ready() noexcept { return true; }
};

template <typename T>
struct ready_future final
{
using value_type        = T;
using raw_const_pointer = T const*;

private:
value_type value_;

public:
__host__ __device__
ready_future() : value_{} {}

ready_future(ready_future&&) = default;
ready_future(ready_future const&) = default;
ready_future& operator=(ready_future&&) = default;
ready_future& operator=(ready_future const&) = default;

template <typename U>
__host__ __device__
explicit ready_future(U&& u) : value_(HYDRA_THRUST_FWD(u)) {}

__host__ __device__
static constexpr bool valid_content() noexcept { return true; }

__host__ __device__
static constexpr bool ready() noexcept { return true; }

__host__ __device__
value_type get() const
{
return value_;
}

HYDRA_THRUST_NODISCARD __host__ __device__
value_type extract() 
{
return std::move(value_);
}

#if defined(HYDRA_THRUST_ENABLE_FUTURE_RAW_DATA_MEMBER)
__host__ __device__
raw_const_pointer data() const
{
return addressof(value_);
}
#endif
};

struct unique_eager_event final
{
protected:
int device_ = 0;
std::unique_ptr<detail::async_signal> async_signal_;

__host__
explicit unique_eager_event(
int device, std::unique_ptr<detail::async_signal> async_signal
)
: device_(device), async_signal_(std::move(async_signal))
{}

public:
__host__
unique_eager_event()
: device_(0), async_signal_()
{}

unique_eager_event(unique_eager_event&&) = default;
unique_eager_event(unique_eager_event const&) = delete;
unique_eager_event& operator=(unique_eager_event&&) = default;
unique_eager_event& operator=(unique_eager_event const&) = delete;

template <typename U>
__host__
explicit unique_eager_event(unique_eager_future<U>&& other)
: device_(other.where()), async_signal_(std::move(other.async_signal_))
{}

__host__
explicit unique_eager_event(new_stream_t const&)
: device_(0)
, async_signal_(new detail::async_signal(detail::unique_stream{}))
{
hydra_thrust::cuda_cub::throw_on_error(cudaGetDevice(&device_));
}

__host__
virtual ~unique_eager_event()
{
if (valid_stream()) wait();
}

__host__
bool valid_stream() const noexcept
{
return bool(async_signal_);
}

__host__
bool ready() const noexcept
{
if (valid_stream())
return stream().ready();
else
return false;
}

__host__
detail::unique_stream& stream()
{
if (!valid_stream())
throw hydra_thrust::event_error(event_errc::no_state);

return async_signal_->stream();
}
detail::unique_stream const& stream() const
{
if (!valid_stream())
throw hydra_thrust::event_error(event_errc::no_state);

return async_signal_->stream();
}

__host__
int where() const noexcept { return device_; }

__host__
void wait()
{
stream().wait();
}

friend __host__
optional<detail::unique_stream>
hydra_thrust::system::cuda::detail::try_acquire_stream(
int device, unique_eager_event& parent
) noexcept;

template <typename... Dependencies>
friend __host__
unique_eager_event
hydra_thrust::system::cuda::detail::make_dependent_event(
std::tuple<Dependencies...>&& deps
);
};

template <typename T>
struct unique_eager_future final
{
HYDRA_THRUST_STATIC_ASSERT_MSG(
(!std::is_same<T, remove_cvref_t<void>>::value)
, "`hydra_thrust::event` should be used to express valueless futures"
);

using value_type        = typename detail::async_value<T>::value_type;
using raw_const_pointer = typename detail::async_value<T>::raw_const_pointer;

private:
int device_ = 0;
std::unique_ptr<detail::async_value<value_type>> async_signal_;

__host__
explicit unique_eager_future(
int device, std::unique_ptr<detail::async_value<value_type>> async_signal
)
: device_(device), async_signal_(std::move(async_signal))
{}

public:
__host__
unique_eager_future()
: device_(0), async_signal_()
{}

unique_eager_future(unique_eager_future&&) = default;
unique_eager_future(unique_eager_future const&) = delete;
unique_eager_future& operator=(unique_eager_future&&) = default;
unique_eager_future& operator=(unique_eager_future const&) = delete;

__host__
explicit unique_eager_future(new_stream_t const&)
: device_(0)
, async_signal_(new detail::async_value<value_type>(detail::unique_stream{}))
{
hydra_thrust::cuda_cub::throw_on_error(cudaGetDevice(&device_));
}

__host__
~unique_eager_future()
{
if (valid_stream()) wait();
}

__host__
bool valid_stream() const noexcept
{
return bool(async_signal_);
}

__host__
bool valid_content() const noexcept
{
if (!valid_stream())
return false;

return async_signal_->valid_content();
}

__host__
bool ready() const noexcept
{
if (valid_stream())
return stream().ready();
else
return false;
}

__host__
detail::unique_stream& stream()
{
if (!valid_stream())
throw hydra_thrust::event_error(event_errc::no_state);

return async_signal_->stream();
}
__host__
detail::unique_stream const& stream() const
{
if (!valid_stream())
throw hydra_thrust::event_error(event_errc::no_state);

return async_signal_->stream();
}

__host__
int where() const noexcept { return device_; }

__host__
void wait()
{
stream().wait();
}

__host__
value_type get()
{
if (!valid_content())
throw hydra_thrust::event_error(event_errc::no_content);

return async_signal_->get();
}

HYDRA_THRUST_NODISCARD __host__
value_type extract()
{
if (!valid_content())
throw hydra_thrust::event_error(event_errc::no_content);

value_type tmp(async_signal_->extract());
async_signal_.reset();
return std::move(tmp);
}

#if defined(HYDRA_THRUST_ENABLE_FUTURE_RAW_DATA_MEMBER)
__host__
raw_const_pointer raw_data() const
{
if (!valid_stream())
throw hydra_thrust::event_error(event_errc::no_state);

return async_signal_->raw_data();
}
#endif

template <typename X>
friend __host__
optional<detail::unique_stream>
hydra_thrust::system::cuda::detail::try_acquire_stream(
int device, unique_eager_future<X>& parent
) noexcept;

template <
typename X, typename XPointer
, typename ComputeContent, typename... Dependencies
>
friend __host__
detail::unique_eager_future_promise_pair<X, XPointer>
hydra_thrust::system::cuda::detail::make_dependent_future(
ComputeContent&& cc, std::tuple<Dependencies...>&& deps
);

friend struct unique_eager_event;
};


namespace detail {

template <typename X, typename Deleter>
__host__
optional<unique_stream>
try_acquire_stream(int, std::unique_ptr<X, Deleter>&) noexcept
{
return {};
}

inline __host__
optional<unique_stream>
try_acquire_stream(int, unique_stream& stream) noexcept
{
return {std::move(stream)};
}

inline __host__
optional<unique_stream>
try_acquire_stream(int, ready_event&) noexcept
{
return {};
}

template <typename X>
__host__
optional<unique_stream>
try_acquire_stream(int, ready_future<X>&) noexcept
{
return {};
}

__host__
optional<unique_stream>
try_acquire_stream(int device, unique_eager_event& parent) noexcept
{
if (parent.valid_stream())
if (device == parent.device_)
return std::move(parent.async_signal_->stream());

return {};
}

template <typename X>
__host__
optional<unique_stream>
try_acquire_stream(int device, unique_eager_future<X>& parent) noexcept
{
if (parent.valid_stream())
if (device == parent.device_)
return std::move(parent.async_signal_->stream());

return {};
}


template <typename... Dependencies>
__host__
acquired_stream acquire_stream_impl(
int, std::tuple<Dependencies...>&, index_sequence<>
) noexcept
{
return {unique_stream{}, {}};
}

template <typename... Dependencies, std::size_t I0, std::size_t... Is>
__host__
acquired_stream acquire_stream_impl(
int device
, std::tuple<Dependencies...>& deps, index_sequence<I0, Is...>
) noexcept
{
auto tr = try_acquire_stream(device, std::get<I0>(deps));

if (tr)
return {std::move(*tr), {I0}};
else
return acquire_stream_impl(device, deps, index_sequence<Is...>{});
}

template <typename... Dependencies>
__host__
acquired_stream acquire_stream(
int device
, std::tuple<Dependencies...>& deps
) noexcept
{
return acquire_stream_impl(
device, deps, make_index_sequence<sizeof...(Dependencies)>{}
);
}


template <typename X, typename Deleter>
__host__
void create_dependency(
unique_stream&, std::unique_ptr<X, Deleter>&
) noexcept
{}

inline __host__
void create_dependency(
unique_stream&, ready_event&
) noexcept
{}

template <typename T>
__host__
void create_dependency(
unique_stream&, ready_future<T>&
) noexcept
{}

inline __host__
void create_dependency(
unique_stream& child, unique_stream& parent
)
{
child.depend_on(parent);
}

inline __host__
void create_dependency(
unique_stream& child, unique_eager_event& parent
)
{
child.depend_on(parent.stream());
}

template <typename X>
__host__
void create_dependency(
unique_stream& child, unique_eager_future<X>& parent
)
{
child.depend_on(parent.stream());
}

template <typename... Dependencies>
__host__
void create_dependencies_impl(
acquired_stream&
, std::tuple<Dependencies...>&, index_sequence<>
)
{}

template <typename... Dependencies, std::size_t I0, std::size_t... Is>
__host__
void create_dependencies_impl(
acquired_stream& as
, std::tuple<Dependencies...>& deps, index_sequence<I0, Is...>
)
{
if (!as.acquired_from || *as.acquired_from != I0)
{
create_dependency(as.stream, std::get<I0>(deps));
}

create_dependencies_impl(as, deps, index_sequence<Is...>{});
}

template <typename... Dependencies>
__host__
void create_dependencies(acquired_stream& as, std::tuple<Dependencies...>& deps)
{
create_dependencies_impl(
as, deps, make_index_sequence<sizeof...(Dependencies)>{}
);
}


template <typename Tuple, typename Indices>
struct find_keep_alives_impl;
template <typename Tuple>
using find_keep_alives
= typename find_keep_alives_impl<
Tuple, make_index_sequence<std::tuple_size<Tuple>::value>
>::type;

template <>
struct find_keep_alives_impl<
std::tuple<>, index_sequence<>
>
{
using type = index_sequence<>;
};

template <
typename... Dependencies
, std::size_t I0, std::size_t... Is
>
struct find_keep_alives_impl<
std::tuple<unique_stream, Dependencies...>, index_sequence<I0, Is...>
>
{
using type = typename find_keep_alives_impl<
std::tuple<Dependencies...>, index_sequence<Is...>
>::type;
};

template <
typename... Dependencies
, std::size_t I0, std::size_t... Is
>
struct find_keep_alives_impl<
std::tuple<ready_event, Dependencies...>, index_sequence<I0, Is...>
>
{
using type = typename find_keep_alives_impl<
std::tuple<Dependencies...>, index_sequence<Is...>
>::type;
};

template <
typename T, typename... Dependencies
, std::size_t I0, std::size_t... Is
>
struct find_keep_alives_impl<
std::tuple<ready_future<T>, Dependencies...>, index_sequence<I0, Is...>
>
{
using type = integer_sequence_push_front<
std::size_t, I0
, typename find_keep_alives_impl<
std::tuple<Dependencies...>, index_sequence<Is...>
>::type
>;
};

template <
typename... Dependencies
, std::size_t I0, std::size_t... Is
>
struct find_keep_alives_impl<
std::tuple<unique_eager_event, Dependencies...>
, index_sequence<I0, Is...>
>
{
using type = integer_sequence_push_front<
std::size_t, I0
, typename find_keep_alives_impl<
std::tuple<Dependencies...>, index_sequence<Is...>
>::type
>;
};

template <
typename X, typename... Dependencies
, std::size_t I0, std::size_t... Is
>
struct find_keep_alives_impl<
std::tuple<unique_eager_future<X>, Dependencies...>
, index_sequence<I0, Is...>
>
{
using type = integer_sequence_push_front<
std::size_t, I0
, typename find_keep_alives_impl<
std::tuple<Dependencies...>, index_sequence<Is...>
>::type
>;
};

template <
typename T, typename Deleter, typename... Dependencies
, std::size_t I0, std::size_t... Is
>
struct find_keep_alives_impl<
std::tuple<std::unique_ptr<T, Deleter>, Dependencies...>
, index_sequence<I0, Is...>
>
{
using type = integer_sequence_push_front<
std::size_t, I0
, typename find_keep_alives_impl<
std::tuple<Dependencies...>, index_sequence<Is...>
>::type
>;
};


template <typename... Dependencies>
__host__
unique_eager_event make_dependent_event(std::tuple<Dependencies...>&& deps)
{
int device = 0;
hydra_thrust::cuda_cub::throw_on_error(cudaGetDevice(&device));

auto as = acquire_stream(device, deps);

create_dependencies(as, deps);

auto ka = tuple_subset(
std::move(deps)
, find_keep_alives<std::tuple<Dependencies...>>{}
);

using async_signal_type = async_keep_alives<decltype(ka)>;

std::unique_ptr<async_signal_type> sig(
new async_signal_type(std::move(as.stream), std::move(ka))
);

return unique_eager_event(device, std::move(sig));
}


template <
typename X, typename XPointer
, typename ComputeContent, typename... Dependencies
>
__host__
unique_eager_future_promise_pair<X, XPointer>
make_dependent_future(ComputeContent&& cc, std::tuple<Dependencies...>&& deps)
{
int device = 0;
hydra_thrust::cuda_cub::throw_on_error(cudaGetDevice(&device));

auto as = acquire_stream(device, deps);

create_dependencies(as, deps);

auto ka = tuple_subset(
std::move(deps)
, find_keep_alives<std::tuple<Dependencies...>>{}
);

using async_signal_type = async_addressable_value_with_keep_alives<
X, XPointer, decltype(ka)
>;

std::unique_ptr<async_signal_type> sig(
new async_signal_type(std::move(as.stream), std::move(ka), std::move(cc))
);

weak_promise<X, XPointer> child_prom(device, sig->data());
unique_eager_future<X> child_fut(device, std::move(sig));

return unique_eager_future_promise_pair<X, XPointer>
{std::move(child_fut), std::move(child_prom)};
}

} 


template <typename... Events>
__host__
unique_eager_event when_all(Events&&... evs)
{
return detail::make_dependent_event(std::make_tuple(std::move(evs)...)); 
}

inline __host__
auto capture_as_dependency(unique_eager_event& dependency)
HYDRA_THRUST_DECLTYPE_RETURNS(std::move(dependency))

template <typename X>
__host__
auto capture_as_dependency(unique_eager_future<X>& dependency)
HYDRA_THRUST_DECLTYPE_RETURNS(std::move(dependency))

}} 

HYDRA_THRUST_END_NS

#endif 

