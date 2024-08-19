


#ifndef CATCH_CONSTRUCTOR_HPP_INCLUDED
#define CATCH_CONSTRUCTOR_HPP_INCLUDED

#include <catch2/internal/catch_move_and_forward.hpp>

#include <type_traits>

namespace Catch {
namespace Benchmark {
namespace Detail {
template <typename T, bool Destruct>
struct ObjectStorage
{
ObjectStorage() = default;

ObjectStorage(const ObjectStorage& other)
{
new(&data) T(other.stored_object());
}

ObjectStorage(ObjectStorage&& other)
{
new(data) T(CATCH_MOVE(other.stored_object()));
}

~ObjectStorage() { destruct_on_exit<T>(); }

template <typename... Args>
void construct(Args&&... args)
{
new (data) T(CATCH_FORWARD(args)...);
}

template <bool AllowManualDestruction = !Destruct>
std::enable_if_t<AllowManualDestruction> destruct()
{
stored_object().~T();
}

private:
template <typename U>
void destruct_on_exit(std::enable_if_t<Destruct, U>* = nullptr) { destruct<true>(); }
template <typename U>
void destruct_on_exit(std::enable_if_t<!Destruct, U>* = nullptr) { }

#if defined( __GNUC__ ) && __GNUC__ <= 6
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
T& stored_object() { return *reinterpret_cast<T*>( data ); }

T const& stored_object() const {
return *reinterpret_cast<T const*>( data );
}
#if defined( __GNUC__ ) && __GNUC__ <= 6
#    pragma GCC diagnostic pop
#endif

alignas( T ) unsigned char data[sizeof( T )]{};
};
} 

template <typename T>
using storage_for = Detail::ObjectStorage<T, true>;

template <typename T>
using destructable_object = Detail::ObjectStorage<T, false>;
} 
} 

#endif 
