



#ifndef BOOST_REGEX_V4_ITERATOR_TRAITS_HPP
#define BOOST_REGEX_V4_ITERATOR_TRAITS_HPP

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4103)
#endif
#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_PREFIX
#endif
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

namespace boost{
namespace BOOST_REGEX_DETAIL_NS{

#if defined(BOOST_NO_STD_ITERATOR_TRAITS)

template <class T>
struct regex_iterator_traits 
{
typedef typename T::iterator_category iterator_category;
typedef typename T::value_type        value_type;
#if !defined(BOOST_NO_STD_ITERATOR)
typedef typename T::difference_type   difference_type;
typedef typename T::pointer           pointer;
typedef typename T::reference         reference;
#else
typedef std::ptrdiff_t                difference_type;
typedef value_type*                   pointer;
typedef value_type&                   reference;
#endif
};

template <class T>
struct pointer_iterator_traits
{
typedef std::ptrdiff_t difference_type;
typedef T value_type;
typedef T* pointer;
typedef T& reference;
typedef std::random_access_iterator_tag iterator_category;
};
template <class T>
struct const_pointer_iterator_traits
{
typedef std::ptrdiff_t difference_type;
typedef T value_type;
typedef const T* pointer;
typedef const T& reference;
typedef std::random_access_iterator_tag iterator_category;
};

template<>
struct regex_iterator_traits<char*> : pointer_iterator_traits<char>{};
template<>
struct regex_iterator_traits<const char*> : const_pointer_iterator_traits<char>{};
template<>
struct regex_iterator_traits<wchar_t*> : pointer_iterator_traits<wchar_t>{};
template<>
struct regex_iterator_traits<const wchar_t*> : const_pointer_iterator_traits<wchar_t>{};
template<>
struct regex_iterator_traits<unsigned char*> : pointer_iterator_traits<char>{};
template<>
struct regex_iterator_traits<const unsigned char*> : const_pointer_iterator_traits<char>{};
template<>
struct regex_iterator_traits<int*> : pointer_iterator_traits<int>{};
template<>
struct regex_iterator_traits<const int*> : const_pointer_iterator_traits<int>{};

#ifdef BOOST_REGEX_HAS_OTHER_WCHAR_T
template<>
struct regex_iterator_traits<unsigned short*> : pointer_iterator_traits<unsigned short>{};
template<>
struct regex_iterator_traits<const unsigned short*> : const_pointer_iterator_traits<unsigned short>{};
#endif

#if defined(__SGI_STL_PORT) && defined(__STL_DEBUG)
template<>
struct regex_iterator_traits<std::string::iterator> : pointer_iterator_traits<char>{};
template<>
struct regex_iterator_traits<std::string::const_iterator> : const_pointer_iterator_traits<char>{};
#ifndef BOOST_NO_STD_WSTRING
template<>
struct regex_iterator_traits<std::wstring::iterator> : pointer_iterator_traits<wchar_t>{};
template<>
struct regex_iterator_traits<std::wstring::const_iterator> : const_pointer_iterator_traits<wchar_t>{};
#endif 
#endif 

#else

template <class T>
struct regex_iterator_traits : public std::iterator_traits<T> {};

#endif

} 
} 

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4103)
#endif
#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_SUFFIX
#endif
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif

