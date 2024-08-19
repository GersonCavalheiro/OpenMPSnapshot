#ifndef BOOST_INTRUSIVE_DETAIL_PARENT_FROM_MEMBER_HPP
#define BOOST_INTRUSIVE_DETAIL_PARENT_FROM_MEMBER_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/detail/workaround.hpp>
#include <cstddef>

#if defined(_MSC_VER)
#define BOOST_INTRUSIVE_MSVC_ABI_PTR_TO_MEMBER
#include <boost/static_assert.hpp>
#endif

namespace boost {
namespace intrusive {
namespace detail {

template<class Parent, class Member>
BOOST_INTRUSIVE_FORCEINLINE std::ptrdiff_t offset_from_pointer_to_member(const Member Parent::* ptr_to_member)
{
#if defined(BOOST_INTRUSIVE_MSVC_ABI_PTR_TO_MEMBER)

union caster_union
{
const Member Parent::* ptr_to_member;
int offset;
} caster;

BOOST_STATIC_ASSERT( sizeof(caster) == sizeof(int) );

caster.ptr_to_member = ptr_to_member;
return std::ptrdiff_t(caster.offset);

#elif defined(__GNUC__)   || defined(__HP_aCC) || defined(BOOST_INTEL) || \
defined(__IBMCPP__) || defined(__DECCXX)
const Parent * const parent = 0;
const char *const member = static_cast<const char*>(static_cast<const void*>(&(parent->*ptr_to_member)));
return std::ptrdiff_t(member - static_cast<const char*>(static_cast<const void*>(parent)));
#else
union caster_union
{
const Member Parent::* ptr_to_member;
std::ptrdiff_t offset;
} caster;
caster.ptr_to_member = ptr_to_member;
return caster.offset - 1;
#endif
}

template<class Parent, class Member>
BOOST_INTRUSIVE_FORCEINLINE Parent *parent_from_member(Member *member, const Member Parent::* ptr_to_member)
{
return static_cast<Parent*>
(
static_cast<void*>
(
static_cast<char*>(static_cast<void*>(member)) - offset_from_pointer_to_member(ptr_to_member)
)
);
}

template<class Parent, class Member>
BOOST_INTRUSIVE_FORCEINLINE const Parent *parent_from_member(const Member *member, const Member Parent::* ptr_to_member)
{
return static_cast<const Parent*>
(
static_cast<const void*>
(
static_cast<const char*>(static_cast<const void*>(member)) - offset_from_pointer_to_member(ptr_to_member)
)
);
}

}  
}  
}  

#include <boost/intrusive/detail/config_end.hpp>

#endif   
