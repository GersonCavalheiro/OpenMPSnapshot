
#ifndef BOOST_TYPE_INDEX_STL_TYPE_INDEX_HPP
#define BOOST_TYPE_INDEX_STL_TYPE_INDEX_HPP


#include <boost/type_index/type_index_facade.hpp>

#if defined(BOOST_NO_RTTI) && !defined(BOOST_MSVC)
#error "File boost/type_index/stl_type_index.ipp is not usable when typeid() is not available."
#endif

#include <typeinfo>
#include <cstring>                                  
#include <stdexcept>
#include <boost/static_assert.hpp>
#include <boost/throw_exception.hpp>
#include <boost/core/demangle.hpp>
#include <boost/type_traits/conditional.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/is_reference.hpp>
#include <boost/type_traits/is_volatile.hpp>
#include <boost/type_traits/remove_cv.hpp>
#include <boost/type_traits/remove_reference.hpp>

#if (defined(_MSC_VER) && _MSC_VER > 1600) \
|| (defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ > 5 && defined(__GXX_EXPERIMENTAL_CXX0X__)) \
|| (defined(__GNUC__) && __GNUC__ > 4 && __cplusplus >= 201103)
#   define BOOST_TYPE_INDEX_STD_TYPE_INDEX_HAS_HASH_CODE
#else
#   include <boost/container_hash/hash.hpp>
#endif

#if (defined(__EDG_VERSION__) && __EDG_VERSION__ < 245) \
|| (defined(__sgi) && defined(_COMPILER_VERSION) && _COMPILER_VERSION <= 744)
#   include <boost/type_traits/is_signed.hpp>
#   include <boost/type_traits/make_signed.hpp>
#   include <boost/type_traits/type_identity.hpp>
#endif

#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

namespace boost { namespace typeindex {

class stl_type_index
: public type_index_facade<
stl_type_index, 
#ifdef BOOST_NO_STD_TYPEINFO
type_info
#else
std::type_info
#endif
> 
{
public:
#ifdef BOOST_NO_STD_TYPEINFO
typedef type_info type_info_t;
#else
typedef std::type_info type_info_t;
#endif

private:
const type_info_t* data_;

public:
inline stl_type_index() BOOST_NOEXCEPT
: data_(&typeid(void))
{}

inline stl_type_index(const type_info_t& data) BOOST_NOEXCEPT
: data_(&data)
{}

inline const type_info_t&  type_info() const BOOST_NOEXCEPT;

inline const char*  raw_name() const BOOST_NOEXCEPT;
inline const char*  name() const BOOST_NOEXCEPT;
inline std::string  pretty_name() const;

inline std::size_t  hash_code() const BOOST_NOEXCEPT;
inline bool         equal(const stl_type_index& rhs) const BOOST_NOEXCEPT;
inline bool         before(const stl_type_index& rhs) const BOOST_NOEXCEPT;

template <class T>
inline static stl_type_index type_id() BOOST_NOEXCEPT;

template <class T>
inline static stl_type_index type_id_with_cvr() BOOST_NOEXCEPT;

template <class T>
inline static stl_type_index type_id_runtime(const T& value) BOOST_NOEXCEPT;
};

inline const stl_type_index::type_info_t& stl_type_index::type_info() const BOOST_NOEXCEPT {
return *data_;
}


inline const char* stl_type_index::raw_name() const BOOST_NOEXCEPT {
#ifdef _MSC_VER
return data_->raw_name();
#else
return data_->name();
#endif
}

inline const char* stl_type_index::name() const BOOST_NOEXCEPT {
return data_->name();
}

inline std::string stl_type_index::pretty_name() const {
static const char cvr_saver_name[] = "boost::typeindex::detail::cvr_saver<";
static BOOST_CONSTEXPR_OR_CONST std::string::size_type cvr_saver_name_len = sizeof(cvr_saver_name) - 1;

const boost::core::scoped_demangled_name demangled_name(data_->name());

const char* begin = demangled_name.get();
if (!begin) {
boost::throw_exception(std::runtime_error("Type name demangling failed"));
}

const std::string::size_type len = std::strlen(begin);
const char* end = begin + len;

if (len > cvr_saver_name_len) {
const char* b = std::strstr(begin, cvr_saver_name);
if (b) {
b += cvr_saver_name_len;

while (*b == ' ') {         
++ b;
}

const char* e = end - 1;
while (e > b && *e != '>') {
-- e;
}

while (e > b && *(e - 1) == ' ') {
-- e;
}

if (b < e) {
begin = b;
end = e;
}
}
}

return std::string(begin, end);
}


inline std::size_t stl_type_index::hash_code() const BOOST_NOEXCEPT {
#ifdef BOOST_TYPE_INDEX_STD_TYPE_INDEX_HAS_HASH_CODE
return data_->hash_code();
#else
return boost::hash_range(raw_name(), raw_name() + std::strlen(raw_name()));
#endif
}



# if (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 5))) \
|| defined(_AIX) \
|| (defined(__sgi) && defined(__host_mips)) \
|| (defined(__hpux) && defined(__HP_aCC)) \
|| (defined(linux) && defined(__INTEL_COMPILER) && defined(__ICC))
#  define BOOST_TYPE_INDEX_CLASSINFO_COMPARE_BY_NAMES
# endif


inline bool stl_type_index::equal(const stl_type_index& rhs) const BOOST_NOEXCEPT {
#ifdef BOOST_TYPE_INDEX_CLASSINFO_COMPARE_BY_NAMES
return raw_name() == rhs.raw_name() || !std::strcmp(raw_name(), rhs.raw_name());
#else
return !!(*data_ == *rhs.data_);
#endif
}

inline bool stl_type_index::before(const stl_type_index& rhs) const BOOST_NOEXCEPT {
#ifdef BOOST_TYPE_INDEX_CLASSINFO_COMPARE_BY_NAMES
return raw_name() != rhs.raw_name() && std::strcmp(raw_name(), rhs.raw_name()) < 0;
#else
return !!data_->before(*rhs.data_);
#endif
}

#undef BOOST_TYPE_INDEX_CLASSINFO_COMPARE_BY_NAMES


template <class T>
inline stl_type_index stl_type_index::type_id() BOOST_NOEXCEPT {
typedef BOOST_DEDUCED_TYPENAME boost::remove_reference<T>::type no_ref_t;
typedef BOOST_DEDUCED_TYPENAME boost::remove_cv<no_ref_t>::type no_cvr_prefinal_t;

#  if (defined(__EDG_VERSION__) && __EDG_VERSION__ < 245) \
|| (defined(__sgi) && defined(_COMPILER_VERSION) && _COMPILER_VERSION <= 744)

typedef BOOST_DEDUCED_TYPENAME boost::conditional<
boost::is_signed<no_cvr_prefinal_t>::value,
boost::make_signed<no_cvr_prefinal_t>,
boost::type_identity<no_cvr_prefinal_t>
>::type no_cvr_prefinal_lazy_t;

typedef BOOST_DEDUCED_TYPENAME no_cvr_prefinal_t::type no_cvr_t;
#else
typedef no_cvr_prefinal_t no_cvr_t;
#endif

return typeid(no_cvr_t);
}

namespace detail {
template <class T> class cvr_saver{};
}

template <class T>
inline stl_type_index stl_type_index::type_id_with_cvr() BOOST_NOEXCEPT {
typedef BOOST_DEDUCED_TYPENAME boost::conditional<
boost::is_reference<T>::value ||  boost::is_const<T>::value || boost::is_volatile<T>::value,
detail::cvr_saver<T>,
T
>::type type;

return typeid(type);
}


template <class T>
inline stl_type_index stl_type_index::type_id_runtime(const T& value) BOOST_NOEXCEPT {
#ifdef BOOST_NO_RTTI
return value.boost_type_index_type_id_runtime_();
#else
return typeid(value);
#endif
}

}} 

#undef BOOST_TYPE_INDEX_STD_TYPE_INDEX_HAS_HASH_CODE

#endif 
