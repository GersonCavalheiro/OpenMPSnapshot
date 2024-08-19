
#ifndef BOOST_TYPE_INDEX_CTTI_TYPE_INDEX_HPP
#define BOOST_TYPE_INDEX_CTTI_TYPE_INDEX_HPP


#include <boost/type_index/type_index_facade.hpp>
#include <boost/type_index/detail/compile_time_type_info.hpp>

#include <cstring>
#include <boost/container_hash/hash.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/remove_cv.hpp>
#include <boost/type_traits/remove_reference.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

namespace boost { namespace typeindex {

namespace detail {


class ctti_data {
#ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
public:
ctti_data() = delete;
ctti_data(const ctti_data&) = delete;
ctti_data& operator=(const ctti_data&) = delete;
#else
private:
ctti_data();
ctti_data(const ctti_data&);
ctti_data& operator=(const ctti_data&);
#endif
};

} 

template <class T>
inline const detail::ctti_data& ctti_construct() BOOST_NOEXCEPT {
return *reinterpret_cast<const detail::ctti_data*>(boost::detail::ctti<T>::n());
}

class ctti_type_index: public type_index_facade<ctti_type_index, detail::ctti_data> {
const char* data_;

inline std::size_t get_raw_name_length() const BOOST_NOEXCEPT;

BOOST_CXX14_CONSTEXPR inline explicit ctti_type_index(const char* data) BOOST_NOEXCEPT
: data_(data)
{}

public:
typedef detail::ctti_data type_info_t;

BOOST_CXX14_CONSTEXPR inline ctti_type_index() BOOST_NOEXCEPT
: data_(boost::detail::ctti<void>::n())
{}

inline ctti_type_index(const type_info_t& data) BOOST_NOEXCEPT
: data_(reinterpret_cast<const char*>(&data))
{}

inline const type_info_t& type_info() const BOOST_NOEXCEPT;
BOOST_CXX14_CONSTEXPR inline const char* raw_name() const BOOST_NOEXCEPT;
BOOST_CXX14_CONSTEXPR inline const char* name() const BOOST_NOEXCEPT;
inline std::string  pretty_name() const;
inline std::size_t  hash_code() const BOOST_NOEXCEPT;

BOOST_CXX14_CONSTEXPR inline bool equal(const ctti_type_index& rhs) const BOOST_NOEXCEPT;
BOOST_CXX14_CONSTEXPR inline bool before(const ctti_type_index& rhs) const BOOST_NOEXCEPT;

template <class T>
BOOST_CXX14_CONSTEXPR inline static ctti_type_index type_id() BOOST_NOEXCEPT;

template <class T>
BOOST_CXX14_CONSTEXPR inline static ctti_type_index type_id_with_cvr() BOOST_NOEXCEPT;

template <class T>
inline static ctti_type_index type_id_runtime(const T& variable) BOOST_NOEXCEPT;
};


inline const ctti_type_index::type_info_t& ctti_type_index::type_info() const BOOST_NOEXCEPT {
return *reinterpret_cast<const detail::ctti_data*>(data_);
}


BOOST_CXX14_CONSTEXPR inline bool ctti_type_index::equal(const ctti_type_index& rhs) const BOOST_NOEXCEPT {
const char* const left = raw_name();
const char* const right = rhs.raw_name();
return  !boost::typeindex::detail::constexpr_strcmp(left, right);
}

BOOST_CXX14_CONSTEXPR inline bool ctti_type_index::before(const ctti_type_index& rhs) const BOOST_NOEXCEPT {
const char* const left = raw_name();
const char* const right = rhs.raw_name();
return  boost::typeindex::detail::constexpr_strcmp(left, right) < 0;
}


template <class T>
BOOST_CXX14_CONSTEXPR inline ctti_type_index ctti_type_index::type_id() BOOST_NOEXCEPT {
typedef BOOST_DEDUCED_TYPENAME boost::remove_reference<T>::type no_ref_t;
typedef BOOST_DEDUCED_TYPENAME boost::remove_cv<no_ref_t>::type no_cvr_t;
return ctti_type_index(boost::detail::ctti<no_cvr_t>::n());
}



template <class T>
BOOST_CXX14_CONSTEXPR inline ctti_type_index ctti_type_index::type_id_with_cvr() BOOST_NOEXCEPT {
return ctti_type_index(boost::detail::ctti<T>::n());
}


template <class T>
inline ctti_type_index ctti_type_index::type_id_runtime(const T& variable) BOOST_NOEXCEPT {
return variable.boost_type_index_type_id_runtime_();
}


BOOST_CXX14_CONSTEXPR inline const char* ctti_type_index::raw_name() const BOOST_NOEXCEPT {
return data_;
}


BOOST_CXX14_CONSTEXPR inline const char* ctti_type_index::name() const BOOST_NOEXCEPT {
return data_;
}

inline std::size_t ctti_type_index::get_raw_name_length() const BOOST_NOEXCEPT {
return std::strlen(raw_name() + detail::ctti_skip_size_at_end);
}


inline std::string ctti_type_index::pretty_name() const {
std::size_t len = get_raw_name_length();
while (raw_name()[len - 1] == ' ') --len; 
return std::string(raw_name(), len);
}


inline std::size_t ctti_type_index::hash_code() const BOOST_NOEXCEPT {
return boost::hash_range(raw_name(), raw_name() + get_raw_name_length());
}


}} 

#endif 

