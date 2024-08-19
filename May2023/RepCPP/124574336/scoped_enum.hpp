


#ifndef BOOST_CORE_SCOPED_ENUM_HPP
#define BOOST_CORE_SCOPED_ENUM_HPP

#include <boost/config.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost
{

#ifdef BOOST_NO_CXX11_SCOPED_ENUMS


template <typename EnumType>
struct native_type
{

typedef typename EnumType::enum_type type;
};


template <typename UnderlyingType, typename EnumType>
inline
BOOST_CONSTEXPR UnderlyingType underlying_cast(EnumType v) BOOST_NOEXCEPT
{
return v.get_underlying_value_();
}


template <typename EnumType>
inline
BOOST_CONSTEXPR typename EnumType::enum_type native_value(EnumType e) BOOST_NOEXCEPT
{
return e.get_native_value_();
}

#else  

template <typename EnumType>
struct native_type
{
typedef EnumType type;
};

template <typename UnderlyingType, typename EnumType>
inline
BOOST_CONSTEXPR UnderlyingType underlying_cast(EnumType v) BOOST_NOEXCEPT
{
return static_cast<UnderlyingType>(v);
}

template <typename EnumType>
inline
BOOST_CONSTEXPR EnumType native_value(EnumType e) BOOST_NOEXCEPT
{
return e;
}

#endif 
}


#ifdef BOOST_NO_CXX11_SCOPED_ENUMS

#ifndef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS

#define BOOST_SCOPED_ENUM_UT_DECLARE_CONVERSION_OPERATOR \
explicit BOOST_CONSTEXPR operator underlying_type() const BOOST_NOEXCEPT { return get_underlying_value_(); }

#else

#define BOOST_SCOPED_ENUM_UT_DECLARE_CONVERSION_OPERATOR

#endif


#define BOOST_SCOPED_ENUM_UT_DECLARE_BEGIN(EnumType, UnderlyingType)    \
struct EnumType {                                                   \
typedef void is_boost_scoped_enum_tag;                          \
typedef UnderlyingType underlying_type;                         \
EnumType() BOOST_NOEXCEPT {}                                    \
explicit BOOST_CONSTEXPR EnumType(underlying_type v) BOOST_NOEXCEPT : v_(v) {}                 \
BOOST_CONSTEXPR underlying_type get_underlying_value_() const BOOST_NOEXCEPT { return v_; }    \
BOOST_SCOPED_ENUM_UT_DECLARE_CONVERSION_OPERATOR                \
private:                                                            \
underlying_type v_;                                             \
typedef EnumType self_type;                                     \
public:                                                             \
enum enum_type

#define BOOST_SCOPED_ENUM_DECLARE_END2() \
BOOST_CONSTEXPR enum_type get_native_value_() const BOOST_NOEXCEPT { return enum_type(v_); } \
friend BOOST_CONSTEXPR bool operator ==(self_type lhs, self_type rhs) BOOST_NOEXCEPT { return enum_type(lhs.v_)==enum_type(rhs.v_); } \
friend BOOST_CONSTEXPR bool operator ==(self_type lhs, enum_type rhs) BOOST_NOEXCEPT { return enum_type(lhs.v_)==rhs; } \
friend BOOST_CONSTEXPR bool operator ==(enum_type lhs, self_type rhs) BOOST_NOEXCEPT { return lhs==enum_type(rhs.v_); } \
friend BOOST_CONSTEXPR bool operator !=(self_type lhs, self_type rhs) BOOST_NOEXCEPT { return enum_type(lhs.v_)!=enum_type(rhs.v_); } \
friend BOOST_CONSTEXPR bool operator !=(self_type lhs, enum_type rhs) BOOST_NOEXCEPT { return enum_type(lhs.v_)!=rhs; } \
friend BOOST_CONSTEXPR bool operator !=(enum_type lhs, self_type rhs) BOOST_NOEXCEPT { return lhs!=enum_type(rhs.v_); } \
friend BOOST_CONSTEXPR bool operator <(self_type lhs, self_type rhs) BOOST_NOEXCEPT { return enum_type(lhs.v_)<enum_type(rhs.v_); } \
friend BOOST_CONSTEXPR bool operator <(self_type lhs, enum_type rhs) BOOST_NOEXCEPT { return enum_type(lhs.v_)<rhs; } \
friend BOOST_CONSTEXPR bool operator <(enum_type lhs, self_type rhs) BOOST_NOEXCEPT { return lhs<enum_type(rhs.v_); } \
friend BOOST_CONSTEXPR bool operator <=(self_type lhs, self_type rhs) BOOST_NOEXCEPT { return enum_type(lhs.v_)<=enum_type(rhs.v_); } \
friend BOOST_CONSTEXPR bool operator <=(self_type lhs, enum_type rhs) BOOST_NOEXCEPT { return enum_type(lhs.v_)<=rhs; } \
friend BOOST_CONSTEXPR bool operator <=(enum_type lhs, self_type rhs) BOOST_NOEXCEPT { return lhs<=enum_type(rhs.v_); } \
friend BOOST_CONSTEXPR bool operator >(self_type lhs, self_type rhs) BOOST_NOEXCEPT { return enum_type(lhs.v_)>enum_type(rhs.v_); } \
friend BOOST_CONSTEXPR bool operator >(self_type lhs, enum_type rhs) BOOST_NOEXCEPT { return enum_type(lhs.v_)>rhs; } \
friend BOOST_CONSTEXPR bool operator >(enum_type lhs, self_type rhs) BOOST_NOEXCEPT { return lhs>enum_type(rhs.v_); } \
friend BOOST_CONSTEXPR bool operator >=(self_type lhs, self_type rhs) BOOST_NOEXCEPT { return enum_type(lhs.v_)>=enum_type(rhs.v_); } \
friend BOOST_CONSTEXPR bool operator >=(self_type lhs, enum_type rhs) BOOST_NOEXCEPT { return enum_type(lhs.v_)>=rhs; } \
friend BOOST_CONSTEXPR bool operator >=(enum_type lhs, self_type rhs) BOOST_NOEXCEPT { return lhs>=enum_type(rhs.v_); } \
};

#define BOOST_SCOPED_ENUM_DECLARE_END(EnumType) \
; \
BOOST_CONSTEXPR EnumType(enum_type v) BOOST_NOEXCEPT : v_(v) {}                 \
BOOST_SCOPED_ENUM_DECLARE_END2()


#define BOOST_SCOPED_ENUM_DECLARE_BEGIN(EnumType) \
BOOST_SCOPED_ENUM_UT_DECLARE_BEGIN(EnumType,int)


#define BOOST_SCOPED_ENUM_NATIVE(EnumType) EnumType::enum_type

#define BOOST_SCOPED_ENUM_FORWARD_DECLARE(EnumType) struct EnumType

#else  

#define BOOST_SCOPED_ENUM_UT_DECLARE_BEGIN(EnumType,UnderlyingType) enum class EnumType : UnderlyingType
#define BOOST_SCOPED_ENUM_DECLARE_BEGIN(EnumType) enum class EnumType
#define BOOST_SCOPED_ENUM_DECLARE_END2()
#define BOOST_SCOPED_ENUM_DECLARE_END(EnumType) ;

#define BOOST_SCOPED_ENUM_NATIVE(EnumType) EnumType
#define BOOST_SCOPED_ENUM_FORWARD_DECLARE(EnumType) enum class EnumType

#endif  

#define BOOST_SCOPED_ENUM_START(name) BOOST_SCOPED_ENUM_DECLARE_BEGIN(name)
#define BOOST_SCOPED_ENUM_END BOOST_SCOPED_ENUM_DECLARE_END2()
#define BOOST_SCOPED_ENUM(name) BOOST_SCOPED_ENUM_NATIVE(name)

#endif  
