#if  defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)) && defined(__x86_64__)
namespace std
{
typedef long unsigned int size_t;
typedef long int ptrdiff_t;
}
namespace std __attribute__ ((__visibility__ ("default")))
{
void
__throw_bad_exception(void) __attribute__((__noreturn__));
void
__throw_bad_alloc(void) __attribute__((__noreturn__));
void
__throw_bad_cast(void) __attribute__((__noreturn__));
void
__throw_bad_typeid(void) __attribute__((__noreturn__));
void
__throw_logic_error(const char*) __attribute__((__noreturn__));
void
__throw_domain_error(const char*) __attribute__((__noreturn__));
void
__throw_invalid_argument(const char*) __attribute__((__noreturn__));
void
__throw_length_error(const char*) __attribute__((__noreturn__));
void
__throw_out_of_range(const char*) __attribute__((__noreturn__));
void
__throw_runtime_error(const char*) __attribute__((__noreturn__));
void
__throw_range_error(const char*) __attribute__((__noreturn__));
void
__throw_overflow_error(const char*) __attribute__((__noreturn__));
void
__throw_underflow_error(const char*) __attribute__((__noreturn__));
void
__throw_ios_failure(const char*) __attribute__((__noreturn__));
void
__throw_system_error(int) __attribute__((__noreturn__));
void
__throw_future_error(int) __attribute__((__noreturn__));
void
__throw_bad_function_call() __attribute__((__noreturn__));
}
namespace __gnu_cxx __attribute__ ((__visibility__ ("default")))
{
template<typename _Iterator, typename _Container>
class __normal_iterator;
}
namespace std __attribute__ ((__visibility__ ("default")))
{
struct __true_type { };
struct __false_type { };
template<bool>
struct __truth_type
{ typedef __false_type __type; };
template<>
struct __truth_type<true>
{ typedef __true_type __type; };
template<class _Sp, class _Tp>
struct __traitor
{
enum { __value = bool(_Sp::__value) || bool(_Tp::__value) };
typedef typename __truth_type<__value>::__type __type;
};
template<typename, typename>
struct __are_same
{
enum { __value = 0 };
typedef __false_type __type;
};
template<typename _Tp>
struct __are_same<_Tp, _Tp>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<typename _Tp>
struct __is_void
{
enum { __value = 0 };
typedef __false_type __type;
};
template<>
struct __is_void<void>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<typename _Tp>
struct __is_integer
{
enum { __value = 0 };
typedef __false_type __type;
};
template<>
struct __is_integer<bool>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<>
struct __is_integer<char>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<>
struct __is_integer<signed char>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<>
struct __is_integer<unsigned char>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<>
struct __is_integer<wchar_t>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<>
struct __is_integer<short>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<>
struct __is_integer<unsigned short>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<>
struct __is_integer<int>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<>
struct __is_integer<unsigned int>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<>
struct __is_integer<long>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<>
struct __is_integer<unsigned long>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<>
struct __is_integer<long long>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<>
struct __is_integer<unsigned long long>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<typename _Tp>
struct __is_floating
{
enum { __value = 0 };
typedef __false_type __type;
};
template<>
struct __is_floating<float>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<>
struct __is_floating<double>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<>
struct __is_floating<long double>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<typename _Tp>
struct __is_pointer
{
enum { __value = 0 };
typedef __false_type __type;
};
template<typename _Tp>
struct __is_pointer<_Tp*>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<typename _Tp>
struct __is_normal_iterator
{
enum { __value = 0 };
typedef __false_type __type;
};
template<typename _Iterator, typename _Container>
struct __is_normal_iterator< __gnu_cxx::__normal_iterator<_Iterator,
_Container> >
{
enum { __value = 1 };
typedef __true_type __type;
};
template<typename _Tp>
struct __is_arithmetic
: public __traitor<__is_integer<_Tp>, __is_floating<_Tp> >
{ };
template<typename _Tp>
struct __is_fundamental
: public __traitor<__is_void<_Tp>, __is_arithmetic<_Tp> >
{ };
template<typename _Tp>
struct __is_scalar
: public __traitor<__is_arithmetic<_Tp>, __is_pointer<_Tp> >
{ };
template<typename _Tp>
struct __is_char
{
enum { __value = 0 };
typedef __false_type __type;
};
template<>
struct __is_char<char>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<>
struct __is_char<wchar_t>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<typename _Tp>
struct __is_byte
{
enum { __value = 0 };
typedef __false_type __type;
};
template<>
struct __is_byte<char>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<>
struct __is_byte<signed char>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<>
struct __is_byte<unsigned char>
{
enum { __value = 1 };
typedef __true_type __type;
};
template<typename _Tp>
struct __is_move_iterator
{
enum { __value = 0 };
typedef __false_type __type;
};
}
namespace __gnu_cxx __attribute__ ((__visibility__ ("default")))
{
template<bool, typename>
struct __enable_if
{ };
template<typename _Tp>
struct __enable_if<true, _Tp>
{ typedef _Tp __type; };
template<bool _Cond, typename _Iftrue, typename _Iffalse>
struct __conditional_type
{ typedef _Iftrue __type; };
template<typename _Iftrue, typename _Iffalse>
struct __conditional_type<false, _Iftrue, _Iffalse>
{ typedef _Iffalse __type; };
template<typename _Tp>
struct __add_unsigned
{
private:
typedef __enable_if<std::__is_integer<_Tp>::__value, _Tp> __if_type;
public:
typedef typename __if_type::__type __type;
};
template<>
struct __add_unsigned<char>
{ typedef unsigned char __type; };
template<>
struct __add_unsigned<signed char>
{ typedef unsigned char __type; };
template<>
struct __add_unsigned<short>
{ typedef unsigned short __type; };
template<>
struct __add_unsigned<int>
{ typedef unsigned int __type; };
template<>
struct __add_unsigned<long>
{ typedef unsigned long __type; };
template<>
struct __add_unsigned<long long>
{ typedef unsigned long long __type; };
template<>
struct __add_unsigned<bool>;
template<>
struct __add_unsigned<wchar_t>;
template<typename _Tp>
struct __remove_unsigned
{
private:
typedef __enable_if<std::__is_integer<_Tp>::__value, _Tp> __if_type;
public:
typedef typename __if_type::__type __type;
};
template<>
struct __remove_unsigned<char>
{ typedef signed char __type; };
template<>
struct __remove_unsigned<unsigned char>
{ typedef signed char __type; };
template<>
struct __remove_unsigned<unsigned short>
{ typedef short __type; };
template<>
struct __remove_unsigned<unsigned int>
{ typedef int __type; };
template<>
struct __remove_unsigned<unsigned long>
{ typedef long __type; };
template<>
struct __remove_unsigned<unsigned long long>
{ typedef long long __type; };
template<>
struct __remove_unsigned<bool>;
template<>
struct __remove_unsigned<wchar_t>;
template<typename _Type>
inline bool
__is_null_pointer(_Type* __ptr)
{ return __ptr == 0; }
template<typename _Type>
inline bool
__is_null_pointer(_Type)
{ return false; }
template<typename _Tp, bool = std::__is_integer<_Tp>::__value>
struct __promote
{ typedef double __type; };
template<typename _Tp>
struct __promote<_Tp, false>
{ };
template<>
struct __promote<long double>
{ typedef long double __type; };
template<>
struct __promote<double>
{ typedef double __type; };
template<>
struct __promote<float>
{ typedef float __type; };
template<typename _Tp, typename _Up,
typename _Tp2 = typename __promote<_Tp>::__type,
typename _Up2 = typename __promote<_Up>::__type>
struct __promote_2
{
typedef __typeof__(_Tp2() + _Up2()) __type;
};
template<typename _Tp, typename _Up, typename _Vp,
typename _Tp2 = typename __promote<_Tp>::__type,
typename _Up2 = typename __promote<_Up>::__type,
typename _Vp2 = typename __promote<_Vp>::__type>
struct __promote_3
{
typedef __typeof__(_Tp2() + _Up2() + _Vp2()) __type;
};
template<typename _Tp, typename _Up, typename _Vp, typename _Wp,
typename _Tp2 = typename __promote<_Tp>::__type,
typename _Up2 = typename __promote<_Up>::__type,
typename _Vp2 = typename __promote<_Vp>::__type,
typename _Wp2 = typename __promote<_Wp>::__type>
struct __promote_4
{
typedef __typeof__(_Tp2() + _Up2() + _Vp2() + _Wp2()) __type;
};
}
namespace __gnu_cxx __attribute__ ((__visibility__ ("default")))
{
template<typename _Value>
struct __numeric_traits_integer
{
static const _Value __min = (((_Value)(-1) < 0) ? (_Value)1 << (sizeof(_Value) * 8 - ((_Value)(-1) < 0)) : (_Value)0);
static const _Value __max = (((_Value)(-1) < 0) ? (((((_Value)1 << ((sizeof(_Value) * 8 - ((_Value)(-1) < 0)) - 1)) - 1) << 1) + 1) : ~(_Value)0);
static const bool __is_signed = ((_Value)(-1) < 0);
static const int __digits = (sizeof(_Value) * 8 - ((_Value)(-1) < 0));
};
template<typename _Value>
const _Value __numeric_traits_integer<_Value>::__min;
template<typename _Value>
const _Value __numeric_traits_integer<_Value>::__max;
template<typename _Value>
const bool __numeric_traits_integer<_Value>::__is_signed;
template<typename _Value>
const int __numeric_traits_integer<_Value>::__digits;
template<typename _Value>
struct __numeric_traits_floating
{
static const int __max_digits10 = (2 + (std::__are_same<_Value, float>::__value ? 24 : std::__are_same<_Value, double>::__value ? 53 : 64) * 643L / 2136);
static const bool __is_signed = true;
static const int __digits10 = (std::__are_same<_Value, float>::__value ? 6 : std::__are_same<_Value, double>::__value ? 15 : 18);
static const int __max_exponent10 = (std::__are_same<_Value, float>::__value ? 38 : std::__are_same<_Value, double>::__value ? 308 : 4932);
};
template<typename _Value>
const int __numeric_traits_floating<_Value>::__max_digits10;
template<typename _Value>
const bool __numeric_traits_floating<_Value>::__is_signed;
template<typename _Value>
const int __numeric_traits_floating<_Value>::__digits10;
template<typename _Value>
const int __numeric_traits_floating<_Value>::__max_exponent10;
template<typename _Value>
struct __numeric_traits
: public __conditional_type<std::__is_integer<_Value>::__value,
__numeric_traits_integer<_Value>,
__numeric_traits_floating<_Value> >::__type
{ };
}
namespace std __attribute__ ((__visibility__ ("default")))
{
template<typename _Tp>
inline _Tp*
__addressof(_Tp& __r)
{
return reinterpret_cast<_Tp*>
(&const_cast<char&>(reinterpret_cast<const volatile char&>(__r)));
}
}
namespace std __attribute__ ((__visibility__ ("default")))
{
template<typename _Tp>
inline void
swap(_Tp& __a, _Tp& __b)
{
_Tp __tmp = (__a);
__a = (__b);
__b = (__tmp);
}
template<typename _Tp, size_t _Nm>
inline void
swap(_Tp (&__a)[_Nm], _Tp (&__b)[_Nm])
{
for (size_t __n = 0; __n < _Nm; ++__n)
swap(__a[__n], __b[__n]);
}
}
namespace std __attribute__ ((__visibility__ ("default")))
{
template<class _T1, class _T2>
struct pair
{
typedef _T1 first_type;
typedef _T2 second_type;
_T1 first;
_T2 second;
pair()
: first(), second() { }
pair(const _T1& __a, const _T2& __b)
: first(__a), second(__b) { }
template<class _U1, class _U2>
pair(const pair<_U1, _U2>& __p)
: first(__p.first), second(__p.second) { }
};
template<class _T1, class _T2>
inline bool
operator==(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{ return __x.first == __y.first && __x.second == __y.second; }
template<class _T1, class _T2>
inline bool
operator<(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{ return __x.first < __y.first
|| (!(__y.first < __x.first) && __x.second < __y.second); }
template<class _T1, class _T2>
inline bool
operator!=(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{ return !(__x == __y); }
template<class _T1, class _T2>
inline bool
operator>(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{ return __y < __x; }
template<class _T1, class _T2>
inline bool
operator<=(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{ return !(__y < __x); }
template<class _T1, class _T2>
inline bool
operator>=(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{ return !(__x < __y); }
template<class _T1, class _T2>
inline pair<_T1, _T2>
make_pair(_T1 __x, _T2 __y)
{ return pair<_T1, _T2>(__x, __y); }
}
namespace std __attribute__ ((__visibility__ ("default")))
{
struct input_iterator_tag { };
struct output_iterator_tag { };
struct forward_iterator_tag : public input_iterator_tag { };
struct bidirectional_iterator_tag : public forward_iterator_tag { };
struct random_access_iterator_tag : public bidirectional_iterator_tag { };
template<typename _Category, typename _Tp, typename _Distance = ptrdiff_t,
typename _Pointer = _Tp*, typename _Reference = _Tp&>
struct iterator
{
typedef _Category iterator_category;
typedef _Tp value_type;
typedef _Distance difference_type;
typedef _Pointer pointer;
typedef _Reference reference;
};
template<typename _Iterator>
struct iterator_traits
{
typedef typename _Iterator::iterator_category iterator_category;
typedef typename _Iterator::value_type value_type;
typedef typename _Iterator::difference_type difference_type;
typedef typename _Iterator::pointer pointer;
typedef typename _Iterator::reference reference;
};
template<typename _Tp>
struct iterator_traits<_Tp*>
{
typedef random_access_iterator_tag iterator_category;
typedef _Tp value_type;
typedef ptrdiff_t difference_type;
typedef _Tp* pointer;
typedef _Tp& reference;
};
template<typename _Tp>
struct iterator_traits<const _Tp*>
{
typedef random_access_iterator_tag iterator_category;
typedef _Tp value_type;
typedef ptrdiff_t difference_type;
typedef const _Tp* pointer;
typedef const _Tp& reference;
};
template<typename _Iter>
inline typename iterator_traits<_Iter>::iterator_category
__iterator_category(const _Iter&)
{ return typename iterator_traits<_Iter>::iterator_category(); }
template<typename _Iterator, bool _HasBase>
struct _Iter_base
{
typedef _Iterator iterator_type;
static iterator_type _S_base(_Iterator __it)
{ return __it; }
};
template<typename _Iterator>
struct _Iter_base<_Iterator, true>
{
typedef typename _Iterator::iterator_type iterator_type;
static iterator_type _S_base(_Iterator __it)
{ return __it.base(); }
};
}
namespace std
{
namespace __debug { }
}
namespace __gnu_debug
{
using namespace std::__debug;
}
namespace std __attribute__ ((__visibility__ ("default")))
{
template<typename _InputIterator>
inline typename iterator_traits<_InputIterator>::difference_type
__distance(_InputIterator __first, _InputIterator __last,
input_iterator_tag)
{
typename iterator_traits<_InputIterator>::difference_type __n = 0;
while (__first != __last)
{
++__first;
++__n;
}
return __n;
}
template<typename _RandomAccessIterator>
inline typename iterator_traits<_RandomAccessIterator>::difference_type
__distance(_RandomAccessIterator __first, _RandomAccessIterator __last,
random_access_iterator_tag)
{
return __last - __first;
}
template<typename _InputIterator>
inline typename iterator_traits<_InputIterator>::difference_type
distance(_InputIterator __first, _InputIterator __last)
{
return std::__distance(__first, __last,
std::__iterator_category(__first));
}
template<typename _InputIterator, typename _Distance>
inline void
__advance(_InputIterator& __i, _Distance __n, input_iterator_tag)
{
;
while (__n--)
++__i;
}
template<typename _BidirectionalIterator, typename _Distance>
inline void
__advance(_BidirectionalIterator& __i, _Distance __n,
bidirectional_iterator_tag)
{
if (__n > 0)
while (__n--)
++__i;
else
while (__n++)
--__i;
}
template<typename _RandomAccessIterator, typename _Distance>
inline void
__advance(_RandomAccessIterator& __i, _Distance __n,
random_access_iterator_tag)
{
__i += __n;
}
template<typename _InputIterator, typename _Distance>
inline void
advance(_InputIterator& __i, _Distance __n)
{
typename iterator_traits<_InputIterator>::difference_type __d = __n;
std::__advance(__i, __d, std::__iterator_category(__i));
}
}
namespace std __attribute__ ((__visibility__ ("default")))
{
template<typename _Iterator>
class reverse_iterator
: public iterator<typename iterator_traits<_Iterator>::iterator_category,
typename iterator_traits<_Iterator>::value_type,
typename iterator_traits<_Iterator>::difference_type,
typename iterator_traits<_Iterator>::pointer,
typename iterator_traits<_Iterator>::reference>
{
protected:
_Iterator current;
typedef iterator_traits<_Iterator> __traits_type;
public:
typedef _Iterator iterator_type;
typedef typename __traits_type::difference_type difference_type;
typedef typename __traits_type::pointer pointer;
typedef typename __traits_type::reference reference;
reverse_iterator() : current() { }
explicit
reverse_iterator(iterator_type __x) : current(__x) { }
reverse_iterator(const reverse_iterator& __x)
: current(__x.current) { }
template<typename _Iter>
reverse_iterator(const reverse_iterator<_Iter>& __x)
: current(__x.base()) { }
iterator_type
base() const
{ return current; }
reference
operator*() const
{
_Iterator __tmp = current;
return *--__tmp;
}
pointer
operator->() const
{ return &(operator*()); }
reverse_iterator&
operator++()
{
--current;
return *this;
}
reverse_iterator
operator++(int)
{
reverse_iterator __tmp = *this;
--current;
return __tmp;
}
reverse_iterator&
operator--()
{
++current;
return *this;
}
reverse_iterator
operator--(int)
{
reverse_iterator __tmp = *this;
++current;
return __tmp;
}
reverse_iterator
operator+(difference_type __n) const
{ return reverse_iterator(current - __n); }
reverse_iterator&
operator+=(difference_type __n)
{
current -= __n;
return *this;
}
reverse_iterator
operator-(difference_type __n) const
{ return reverse_iterator(current + __n); }
reverse_iterator&
operator-=(difference_type __n)
{
current += __n;
return *this;
}
reference
operator[](difference_type __n) const
{ return *(*this + __n); }
};
template<typename _Iterator>
inline bool
operator==(const reverse_iterator<_Iterator>& __x,
const reverse_iterator<_Iterator>& __y)
{ return __x.base() == __y.base(); }
template<typename _Iterator>
inline bool
operator<(const reverse_iterator<_Iterator>& __x,
const reverse_iterator<_Iterator>& __y)
{ return __y.base() < __x.base(); }
template<typename _Iterator>
inline bool
operator!=(const reverse_iterator<_Iterator>& __x,
const reverse_iterator<_Iterator>& __y)
{ return !(__x == __y); }
template<typename _Iterator>
inline bool
operator>(const reverse_iterator<_Iterator>& __x,
const reverse_iterator<_Iterator>& __y)
{ return __y < __x; }
template<typename _Iterator>
inline bool
operator<=(const reverse_iterator<_Iterator>& __x,
const reverse_iterator<_Iterator>& __y)
{ return !(__y < __x); }
template<typename _Iterator>
inline bool
operator>=(const reverse_iterator<_Iterator>& __x,
const reverse_iterator<_Iterator>& __y)
{ return !(__x < __y); }
template<typename _Iterator>
inline typename reverse_iterator<_Iterator>::difference_type
operator-(const reverse_iterator<_Iterator>& __x,
const reverse_iterator<_Iterator>& __y)
{ return __y.base() - __x.base(); }
template<typename _Iterator>
inline reverse_iterator<_Iterator>
operator+(typename reverse_iterator<_Iterator>::difference_type __n,
const reverse_iterator<_Iterator>& __x)
{ return reverse_iterator<_Iterator>(__x.base() - __n); }
template<typename _IteratorL, typename _IteratorR>
inline bool
operator==(const reverse_iterator<_IteratorL>& __x,
const reverse_iterator<_IteratorR>& __y)
{ return __x.base() == __y.base(); }
template<typename _IteratorL, typename _IteratorR>
inline bool
operator<(const reverse_iterator<_IteratorL>& __x,
const reverse_iterator<_IteratorR>& __y)
{ return __y.base() < __x.base(); }
template<typename _IteratorL, typename _IteratorR>
inline bool
operator!=(const reverse_iterator<_IteratorL>& __x,
const reverse_iterator<_IteratorR>& __y)
{ return !(__x == __y); }
template<typename _IteratorL, typename _IteratorR>
inline bool
operator>(const reverse_iterator<_IteratorL>& __x,
const reverse_iterator<_IteratorR>& __y)
{ return __y < __x; }
template<typename _IteratorL, typename _IteratorR>
inline bool
operator<=(const reverse_iterator<_IteratorL>& __x,
const reverse_iterator<_IteratorR>& __y)
{ return !(__y < __x); }
template<typename _IteratorL, typename _IteratorR>
inline bool
operator>=(const reverse_iterator<_IteratorL>& __x,
const reverse_iterator<_IteratorR>& __y)
{ return !(__x < __y); }
template<typename _IteratorL, typename _IteratorR>
inline typename reverse_iterator<_IteratorL>::difference_type
operator-(const reverse_iterator<_IteratorL>& __x,
const reverse_iterator<_IteratorR>& __y)
{ return __y.base() - __x.base(); }
template<typename _Container>
class back_insert_iterator
: public iterator<output_iterator_tag, void, void, void, void>
{
protected:
_Container* container;
public:
typedef _Container container_type;
explicit
back_insert_iterator(_Container& __x) : container(&__x) { }
back_insert_iterator&
operator=(typename _Container::const_reference __value)
{
container->push_back(__value);
return *this;
}
back_insert_iterator&
operator*()
{ return *this; }
back_insert_iterator&
operator++()
{ return *this; }
back_insert_iterator
operator++(int)
{ return *this; }
};
template<typename _Container>
inline back_insert_iterator<_Container>
back_inserter(_Container& __x)
{ return back_insert_iterator<_Container>(__x); }
template<typename _Container>
class front_insert_iterator
: public iterator<output_iterator_tag, void, void, void, void>
{
protected:
_Container* container;
public:
typedef _Container container_type;
explicit front_insert_iterator(_Container& __x) : container(&__x) { }
front_insert_iterator&
operator=(typename _Container::const_reference __value)
{
container->push_front(__value);
return *this;
}
front_insert_iterator&
operator*()
{ return *this; }
front_insert_iterator&
operator++()
{ return *this; }
front_insert_iterator
operator++(int)
{ return *this; }
};
template<typename _Container>
inline front_insert_iterator<_Container>
front_inserter(_Container& __x)
{ return front_insert_iterator<_Container>(__x); }
template<typename _Container>
class insert_iterator
: public iterator<output_iterator_tag, void, void, void, void>
{
protected:
_Container* container;
typename _Container::iterator iter;
public:
typedef _Container container_type;
insert_iterator(_Container& __x, typename _Container::iterator __i)
: container(&__x), iter(__i) {}
insert_iterator&
operator=(typename _Container::const_reference __value)
{
iter = container->insert(iter, __value);
++iter;
return *this;
}
insert_iterator&
operator*()
{ return *this; }
insert_iterator&
operator++()
{ return *this; }
insert_iterator&
operator++(int)
{ return *this; }
};
template<typename _Container, typename _Iterator>
inline insert_iterator<_Container>
inserter(_Container& __x, _Iterator __i)
{
return insert_iterator<_Container>(__x,
typename _Container::iterator(__i));
}
}
namespace __gnu_cxx __attribute__ ((__visibility__ ("default")))
{
using std::iterator_traits;
using std::iterator;
template<typename _Iterator, typename _Container>
class __normal_iterator
{
protected:
_Iterator _M_current;
typedef iterator_traits<_Iterator> __traits_type;
public:
typedef _Iterator iterator_type;
typedef typename __traits_type::iterator_category iterator_category;
typedef typename __traits_type::value_type value_type;
typedef typename __traits_type::difference_type difference_type;
typedef typename __traits_type::reference reference;
typedef typename __traits_type::pointer pointer;
__normal_iterator() : _M_current(_Iterator()) { }
explicit
__normal_iterator(const _Iterator& __i) : _M_current(__i) { }
template<typename _Iter>
__normal_iterator(const __normal_iterator<_Iter,
typename __enable_if<
(std::__are_same<_Iter, typename _Container::pointer>::__value),
_Container>::__type>& __i)
: _M_current(__i.base()) { }
reference
operator*() const
{ return *_M_current; }
pointer
operator->() const
{ return _M_current; }
__normal_iterator&
operator++()
{
++_M_current;
return *this;
}
__normal_iterator
operator++(int)
{ return __normal_iterator(_M_current++); }
__normal_iterator&
operator--()
{
--_M_current;
return *this;
}
__normal_iterator
operator--(int)
{ return __normal_iterator(_M_current--); }
reference
operator[](const difference_type& __n) const
{ return _M_current[__n]; }
__normal_iterator&
operator+=(const difference_type& __n)
{ _M_current += __n; return *this; }
__normal_iterator
operator+(const difference_type& __n) const
{ return __normal_iterator(_M_current + __n); }
__normal_iterator&
operator-=(const difference_type& __n)
{ _M_current -= __n; return *this; }
__normal_iterator
operator-(const difference_type& __n) const
{ return __normal_iterator(_M_current - __n); }
const _Iterator&
base() const
{ return _M_current; }
};
template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator==(const __normal_iterator<_IteratorL, _Container>& __lhs,
const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() == __rhs.base(); }
template<typename _Iterator, typename _Container>
inline bool
operator==(const __normal_iterator<_Iterator, _Container>& __lhs,
const __normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() == __rhs.base(); }
template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator!=(const __normal_iterator<_IteratorL, _Container>& __lhs,
const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() != __rhs.base(); }
template<typename _Iterator, typename _Container>
inline bool
operator!=(const __normal_iterator<_Iterator, _Container>& __lhs,
const __normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() != __rhs.base(); }
template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator<(const __normal_iterator<_IteratorL, _Container>& __lhs,
const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() < __rhs.base(); }
template<typename _Iterator, typename _Container>
inline bool
operator<(const __normal_iterator<_Iterator, _Container>& __lhs,
const __normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() < __rhs.base(); }
template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator>(const __normal_iterator<_IteratorL, _Container>& __lhs,
const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() > __rhs.base(); }
template<typename _Iterator, typename _Container>
inline bool
operator>(const __normal_iterator<_Iterator, _Container>& __lhs,
const __normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() > __rhs.base(); }
template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator<=(const __normal_iterator<_IteratorL, _Container>& __lhs,
const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() <= __rhs.base(); }
template<typename _Iterator, typename _Container>
inline bool
operator<=(const __normal_iterator<_Iterator, _Container>& __lhs,
const __normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() <= __rhs.base(); }
template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator>=(const __normal_iterator<_IteratorL, _Container>& __lhs,
const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() >= __rhs.base(); }
template<typename _Iterator, typename _Container>
inline bool
operator>=(const __normal_iterator<_Iterator, _Container>& __lhs,
const __normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() >= __rhs.base(); }
template<typename _IteratorL, typename _IteratorR, typename _Container>
inline typename __normal_iterator<_IteratorL, _Container>::difference_type
operator-(const __normal_iterator<_IteratorL, _Container>& __lhs,
const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() - __rhs.base(); }
template<typename _Iterator, typename _Container>
inline typename __normal_iterator<_Iterator, _Container>::difference_type
operator-(const __normal_iterator<_Iterator, _Container>& __lhs,
const __normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() - __rhs.base(); }
template<typename _Iterator, typename _Container>
inline __normal_iterator<_Iterator, _Container>
operator+(typename __normal_iterator<_Iterator, _Container>::difference_type
__n, const __normal_iterator<_Iterator, _Container>& __i)
{ return __normal_iterator<_Iterator, _Container>(__i.base() + __n); }
}
namespace std __attribute__ ((__visibility__ ("default")))
{
template<bool _BoolType>
struct __iter_swap
{
template<typename _ForwardIterator1, typename _ForwardIterator2>
static void
iter_swap(_ForwardIterator1 __a, _ForwardIterator2 __b)
{
typedef typename iterator_traits<_ForwardIterator1>::value_type
_ValueType1;
_ValueType1 __tmp = (*__a);
*__a = (*__b);
*__b = (__tmp);
}
};
template<>
struct __iter_swap<true>
{
template<typename _ForwardIterator1, typename _ForwardIterator2>
static void
iter_swap(_ForwardIterator1 __a, _ForwardIterator2 __b)
{
swap(*__a, *__b);
}
};
template<typename _ForwardIterator1, typename _ForwardIterator2>
inline void
iter_swap(_ForwardIterator1 __a, _ForwardIterator2 __b)
{
typedef typename iterator_traits<_ForwardIterator1>::value_type
_ValueType1;
typedef typename iterator_traits<_ForwardIterator2>::value_type
_ValueType2;
typedef typename iterator_traits<_ForwardIterator1>::reference
_ReferenceType1;
typedef typename iterator_traits<_ForwardIterator2>::reference
_ReferenceType2;
std::__iter_swap<__are_same<_ValueType1, _ValueType2>::__value
&& __are_same<_ValueType1&, _ReferenceType1>::__value
&& __are_same<_ValueType2&, _ReferenceType2>::__value>::
iter_swap(__a, __b);
}
template<typename _ForwardIterator1, typename _ForwardIterator2>
_ForwardIterator2
swap_ranges(_ForwardIterator1 __first1, _ForwardIterator1 __last1,
_ForwardIterator2 __first2)
{
;
for (; __first1 != __last1; ++__first1, ++__first2)
std::iter_swap(__first1, __first2);
return __first2;
}
template<typename _Tp>
inline const _Tp&
min(const _Tp& __a, const _Tp& __b)
{
if (__b < __a)
return __b;
return __a;
}
template<typename _Tp>
inline const _Tp&
max(const _Tp& __a, const _Tp& __b)
{
if (__a < __b)
return __b;
return __a;
}
template<typename _Tp, typename _Compare>
inline const _Tp&
min(const _Tp& __a, const _Tp& __b, _Compare __comp)
{
if (__comp(__b, __a))
return __b;
return __a;
}
template<typename _Tp, typename _Compare>
inline const _Tp&
max(const _Tp& __a, const _Tp& __b, _Compare __comp)
{
if (__comp(__a, __b))
return __b;
return __a;
}
template<typename _Iterator>
struct _Niter_base
: _Iter_base<_Iterator, __is_normal_iterator<_Iterator>::__value>
{ };
template<typename _Iterator>
inline typename _Niter_base<_Iterator>::iterator_type
__niter_base(_Iterator __it)
{ return std::_Niter_base<_Iterator>::_S_base(__it); }
template<typename _Iterator>
struct _Miter_base
: _Iter_base<_Iterator, __is_move_iterator<_Iterator>::__value>
{ };
template<typename _Iterator>
inline typename _Miter_base<_Iterator>::iterator_type
__miter_base(_Iterator __it)
{ return std::_Miter_base<_Iterator>::_S_base(__it); }
template<bool, bool, typename>
struct __copy_move
{
template<typename _II, typename _OI>
static _OI
__copy_m(_II __first, _II __last, _OI __result)
{
for (; __first != __last; ++__result, ++__first)
*__result = *__first;
return __result;
}
};
template<>
struct __copy_move<false, false, random_access_iterator_tag>
{
template<typename _II, typename _OI>
static _OI
__copy_m(_II __first, _II __last, _OI __result)
{
typedef typename iterator_traits<_II>::difference_type _Distance;
for(_Distance __n = __last - __first; __n > 0; --__n)
{
*__result = *__first;
++__first;
++__result;
}
return __result;
}
};
template<bool _IsMove>
struct __copy_move<_IsMove, true, random_access_iterator_tag>
{
template<typename _Tp>
static _Tp*
__copy_m(const _Tp* __first, const _Tp* __last, _Tp* __result)
{
const ptrdiff_t _Num = __last - __first;
if (_Num)
__builtin_memmove(__result, __first, sizeof(_Tp) * _Num);
return __result + _Num;
}
};
template<bool _IsMove, typename _II, typename _OI>
inline _OI
__copy_move_a(_II __first, _II __last, _OI __result)
{
typedef typename iterator_traits<_II>::value_type _ValueTypeI;
typedef typename iterator_traits<_OI>::value_type _ValueTypeO;
typedef typename iterator_traits<_II>::iterator_category _Category;
const bool __simple = (__is_trivial(_ValueTypeI)
&& __is_pointer<_II>::__value
&& __is_pointer<_OI>::__value
&& __are_same<_ValueTypeI, _ValueTypeO>::__value);
return std::__copy_move<_IsMove, __simple,
_Category>::__copy_m(__first, __last, __result);
}
template<typename _CharT>
struct char_traits;
template<typename _CharT, typename _Traits>
class istreambuf_iterator;
template<typename _CharT, typename _Traits>
class ostreambuf_iterator;
template<bool _IsMove, typename _CharT>
typename __gnu_cxx::__enable_if<__is_char<_CharT>::__value,
ostreambuf_iterator<_CharT, char_traits<_CharT> > >::__type
__copy_move_a2(_CharT*, _CharT*,
ostreambuf_iterator<_CharT, char_traits<_CharT> >);
template<bool _IsMove, typename _CharT>
typename __gnu_cxx::__enable_if<__is_char<_CharT>::__value,
ostreambuf_iterator<_CharT, char_traits<_CharT> > >::__type
__copy_move_a2(const _CharT*, const _CharT*,
ostreambuf_iterator<_CharT, char_traits<_CharT> >);
template<bool _IsMove, typename _CharT>
typename __gnu_cxx::__enable_if<__is_char<_CharT>::__value,
_CharT*>::__type
__copy_move_a2(istreambuf_iterator<_CharT, char_traits<_CharT> >,
istreambuf_iterator<_CharT, char_traits<_CharT> >, _CharT*);
template<bool _IsMove, typename _II, typename _OI>
inline _OI
__copy_move_a2(_II __first, _II __last, _OI __result)
{
return _OI(std::__copy_move_a<_IsMove>(std::__niter_base(__first),
std::__niter_base(__last),
std::__niter_base(__result)));
}
template<typename _II, typename _OI>
inline _OI
copy(_II __first, _II __last, _OI __result)
{
;
return (std::__copy_move_a2<__is_move_iterator<_II>::__value>
(std::__miter_base(__first), std::__miter_base(__last),
__result));
}
template<bool, bool, typename>
struct __copy_move_backward
{
template<typename _BI1, typename _BI2>
static _BI2
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result)
{
while (__first != __last)
*--__result = *--__last;
return __result;
}
};
template<>
struct __copy_move_backward<false, false, random_access_iterator_tag>
{
template<typename _BI1, typename _BI2>
static _BI2
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result)
{
typename iterator_traits<_BI1>::difference_type __n;
for (__n = __last - __first; __n > 0; --__n)
*--__result = *--__last;
return __result;
}
};
template<bool _IsMove>
struct __copy_move_backward<_IsMove, true, random_access_iterator_tag>
{
template<typename _Tp>
static _Tp*
__copy_move_b(const _Tp* __first, const _Tp* __last, _Tp* __result)
{
const ptrdiff_t _Num = __last - __first;
if (_Num)
__builtin_memmove(__result - _Num, __first, sizeof(_Tp) * _Num);
return __result - _Num;
}
};
template<bool _IsMove, typename _BI1, typename _BI2>
inline _BI2
__copy_move_backward_a(_BI1 __first, _BI1 __last, _BI2 __result)
{
typedef typename iterator_traits<_BI1>::value_type _ValueType1;
typedef typename iterator_traits<_BI2>::value_type _ValueType2;
typedef typename iterator_traits<_BI1>::iterator_category _Category;
const bool __simple = (__is_trivial(_ValueType1)
&& __is_pointer<_BI1>::__value
&& __is_pointer<_BI2>::__value
&& __are_same<_ValueType1, _ValueType2>::__value);
return std::__copy_move_backward<_IsMove, __simple,
_Category>::__copy_move_b(__first,
__last,
__result);
}
template<bool _IsMove, typename _BI1, typename _BI2>
inline _BI2
__copy_move_backward_a2(_BI1 __first, _BI1 __last, _BI2 __result)
{
return _BI2(std::__copy_move_backward_a<_IsMove>
(std::__niter_base(__first), std::__niter_base(__last),
std::__niter_base(__result)));
}
template<typename _BI1, typename _BI2>
inline _BI2
copy_backward(_BI1 __first, _BI1 __last, _BI2 __result)
{
;
return (std::__copy_move_backward_a2<__is_move_iterator<_BI1>::__value>
(std::__miter_base(__first), std::__miter_base(__last),
__result));
}
template<typename _ForwardIterator, typename _Tp>
inline typename
__gnu_cxx::__enable_if<!__is_scalar<_Tp>::__value, void>::__type
__fill_a(_ForwardIterator __first, _ForwardIterator __last,
const _Tp& __value)
{
for (; __first != __last; ++__first)
*__first = __value;
}
template<typename _ForwardIterator, typename _Tp>
inline typename
__gnu_cxx::__enable_if<__is_scalar<_Tp>::__value, void>::__type
__fill_a(_ForwardIterator __first, _ForwardIterator __last,
const _Tp& __value)
{
const _Tp __tmp = __value;
for (; __first != __last; ++__first)
*__first = __tmp;
}
template<typename _Tp>
inline typename
__gnu_cxx::__enable_if<__is_byte<_Tp>::__value, void>::__type
__fill_a(_Tp* __first, _Tp* __last, const _Tp& __c)
{
const _Tp __tmp = __c;
__builtin_memset(__first, static_cast<unsigned char>(__tmp),
__last - __first);
}
template<typename _ForwardIterator, typename _Tp>
inline void
fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
;
std::__fill_a(std::__niter_base(__first), std::__niter_base(__last),
__value);
}
template<typename _OutputIterator, typename _Size, typename _Tp>
inline typename
__gnu_cxx::__enable_if<!__is_scalar<_Tp>::__value, _OutputIterator>::__type
__fill_n_a(_OutputIterator __first, _Size __n, const _Tp& __value)
{
for (__decltype(__n + 0) __niter = __n;
__niter > 0; --__niter, ++__first)
*__first = __value;
return __first;
}
template<typename _OutputIterator, typename _Size, typename _Tp>
inline typename
__gnu_cxx::__enable_if<__is_scalar<_Tp>::__value, _OutputIterator>::__type
__fill_n_a(_OutputIterator __first, _Size __n, const _Tp& __value)
{
const _Tp __tmp = __value;
for (__decltype(__n + 0) __niter = __n;
__niter > 0; --__niter, ++__first)
*__first = __tmp;
return __first;
}
template<typename _Size, typename _Tp>
inline typename
__gnu_cxx::__enable_if<__is_byte<_Tp>::__value, _Tp*>::__type
__fill_n_a(_Tp* __first, _Size __n, const _Tp& __c)
{
std::__fill_a(__first, __first + __n, __c);
return __first + __n;
}
template<typename _OI, typename _Size, typename _Tp>
inline _OI
fill_n(_OI __first, _Size __n, const _Tp& __value)
{
return _OI(std::__fill_n_a(std::__niter_base(__first), __n, __value));
}
template<bool _BoolType>
struct __equal
{
template<typename _II1, typename _II2>
static bool
equal(_II1 __first1, _II1 __last1, _II2 __first2)
{
for (; __first1 != __last1; ++__first1, ++__first2)
if (!(*__first1 == *__first2))
return false;
return true;
}
};
template<>
struct __equal<true>
{
template<typename _Tp>
static bool
equal(const _Tp* __first1, const _Tp* __last1, const _Tp* __first2)
{
return !__builtin_memcmp(__first1, __first2, sizeof(_Tp)
* (__last1 - __first1));
}
};
template<typename _II1, typename _II2>
inline bool
__equal_aux(_II1 __first1, _II1 __last1, _II2 __first2)
{
typedef typename iterator_traits<_II1>::value_type _ValueType1;
typedef typename iterator_traits<_II2>::value_type _ValueType2;
const bool __simple = ((__is_integer<_ValueType1>::__value
|| __is_pointer<_ValueType1>::__value)
&& __is_pointer<_II1>::__value
&& __is_pointer<_II2>::__value
&& __are_same<_ValueType1, _ValueType2>::__value);
return std::__equal<__simple>::equal(__first1, __last1, __first2);
}
template<typename, typename>
struct __lc_rai
{
template<typename _II1, typename _II2>
static _II1
__newlast1(_II1, _II1 __last1, _II2, _II2)
{ return __last1; }
template<typename _II>
static bool
__cnd2(_II __first, _II __last)
{ return __first != __last; }
};
template<>
struct __lc_rai<random_access_iterator_tag, random_access_iterator_tag>
{
template<typename _RAI1, typename _RAI2>
static _RAI1
__newlast1(_RAI1 __first1, _RAI1 __last1,
_RAI2 __first2, _RAI2 __last2)
{
const typename iterator_traits<_RAI1>::difference_type
__diff1 = __last1 - __first1;
const typename iterator_traits<_RAI2>::difference_type
__diff2 = __last2 - __first2;
return __diff2 < __diff1 ? __first1 + __diff2 : __last1;
}
template<typename _RAI>
static bool
__cnd2(_RAI, _RAI)
{ return true; }
};
template<bool _BoolType>
struct __lexicographical_compare
{
template<typename _II1, typename _II2>
static bool __lc(_II1, _II1, _II2, _II2);
};
template<bool _BoolType>
template<typename _II1, typename _II2>
bool
__lexicographical_compare<_BoolType>::
__lc(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2)
{
typedef typename iterator_traits<_II1>::iterator_category _Category1;
typedef typename iterator_traits<_II2>::iterator_category _Category2;
typedef std::__lc_rai<_Category1, _Category2> __rai_type;
__last1 = __rai_type::__newlast1(__first1, __last1,
__first2, __last2);
for (; __first1 != __last1 && __rai_type::__cnd2(__first2, __last2);
++__first1, ++__first2)
{
if (*__first1 < *__first2)
return true;
if (*__first2 < *__first1)
return false;
}
return __first1 == __last1 && __first2 != __last2;
}
template<>
struct __lexicographical_compare<true>
{
template<typename _Tp, typename _Up>
static bool
__lc(const _Tp* __first1, const _Tp* __last1,
const _Up* __first2, const _Up* __last2)
{
const size_t __len1 = __last1 - __first1;
const size_t __len2 = __last2 - __first2;
const int __result = __builtin_memcmp(__first1, __first2,
std::min(__len1, __len2));
return __result != 0 ? __result < 0 : __len1 < __len2;
}
};
template<typename _II1, typename _II2>
inline bool
__lexicographical_compare_aux(_II1 __first1, _II1 __last1,
_II2 __first2, _II2 __last2)
{
typedef typename iterator_traits<_II1>::value_type _ValueType1;
typedef typename iterator_traits<_II2>::value_type _ValueType2;
const bool __simple =
(__is_byte<_ValueType1>::__value && __is_byte<_ValueType2>::__value
&& !__gnu_cxx::__numeric_traits<_ValueType1>::__is_signed
&& !__gnu_cxx::__numeric_traits<_ValueType2>::__is_signed
&& __is_pointer<_II1>::__value
&& __is_pointer<_II2>::__value);
return std::__lexicographical_compare<__simple>::__lc(__first1, __last1,
__first2, __last2);
}
template<typename _ForwardIterator, typename _Tp>
_ForwardIterator
lower_bound(_ForwardIterator __first, _ForwardIterator __last,
const _Tp& __val)
{
typedef typename iterator_traits<_ForwardIterator>::difference_type
_DistanceType;
;
_DistanceType __len = std::distance(__first, __last);
while (__len > 0)
{
_DistanceType __half = __len >> 1;
_ForwardIterator __middle = __first;
std::advance(__middle, __half);
if (*__middle < __val)
{
__first = __middle;
++__first;
__len = __len - __half - 1;
}
else
__len = __half;
}
return __first;
}
inline int
__lg(int __n)
{ return sizeof(int) * 8 - 1 - __builtin_clz(__n); }
inline unsigned
__lg(unsigned __n)
{ return sizeof(int) * 8 - 1 - __builtin_clz(__n); }
inline long
__lg(long __n)
{ return sizeof(long) * 8 - 1 - __builtin_clzl(__n); }
inline unsigned long
__lg(unsigned long __n)
{ return sizeof(long) * 8 - 1 - __builtin_clzl(__n); }
inline long long
__lg(long long __n)
{ return sizeof(long long) * 8 - 1 - __builtin_clzll(__n); }
inline unsigned long long
__lg(unsigned long long __n)
{ return sizeof(long long) * 8 - 1 - __builtin_clzll(__n); }
template<typename _II1, typename _II2>
inline bool
equal(_II1 __first1, _II1 __last1, _II2 __first2)
{
;
return std::__equal_aux(std::__niter_base(__first1),
std::__niter_base(__last1),
std::__niter_base(__first2));
}
template<typename _IIter1, typename _IIter2, typename _BinaryPredicate>
inline bool
equal(_IIter1 __first1, _IIter1 __last1,
_IIter2 __first2, _BinaryPredicate __binary_pred)
{
;
for (; __first1 != __last1; ++__first1, ++__first2)
if (!bool(__binary_pred(*__first1, *__first2)))
return false;
return true;
}
template<typename _II1, typename _II2>
inline bool
lexicographical_compare(_II1 __first1, _II1 __last1,
_II2 __first2, _II2 __last2)
{
;
;
return std::__lexicographical_compare_aux(std::__niter_base(__first1),
std::__niter_base(__last1),
std::__niter_base(__first2),
std::__niter_base(__last2));
}
template<typename _II1, typename _II2, typename _Compare>
bool
lexicographical_compare(_II1 __first1, _II1 __last1,
_II2 __first2, _II2 __last2, _Compare __comp)
{
typedef typename iterator_traits<_II1>::iterator_category _Category1;
typedef typename iterator_traits<_II2>::iterator_category _Category2;
typedef std::__lc_rai<_Category1, _Category2> __rai_type;
;
;
__last1 = __rai_type::__newlast1(__first1, __last1, __first2, __last2);
for (; __first1 != __last1 && __rai_type::__cnd2(__first2, __last2);
++__first1, ++__first2)
{
if (__comp(*__first1, *__first2))
return true;
if (__comp(*__first2, *__first1))
return false;
}
return __first1 == __last1 && __first2 != __last2;
}
template<typename _InputIterator1, typename _InputIterator2>
pair<_InputIterator1, _InputIterator2>
mismatch(_InputIterator1 __first1, _InputIterator1 __last1,
_InputIterator2 __first2)
{
;
while (__first1 != __last1 && *__first1 == *__first2)
{
++__first1;
++__first2;
}
return pair<_InputIterator1, _InputIterator2>(__first1, __first2);
}
template<typename _InputIterator1, typename _InputIterator2,
typename _BinaryPredicate>
pair<_InputIterator1, _InputIterator2>
mismatch(_InputIterator1 __first1, _InputIterator1 __last1,
_InputIterator2 __first2, _BinaryPredicate __binary_pred)
{
;
while (__first1 != __last1 && bool(__binary_pred(*__first1, *__first2)))
{
++__first1;
++__first2;
}
return pair<_InputIterator1, _InputIterator2>(__first1, __first2);
}
}
#pragma GCC visibility push(default)
extern "C++" {
namespace std
{
class exception
{
public:
exception() throw() { }
virtual ~exception() throw();
virtual const char* what() const throw();
};
class bad_exception : public exception
{
public:
bad_exception() throw() { }
virtual ~bad_exception() throw();
virtual const char* what() const throw();
};
typedef void (*terminate_handler) ();
typedef void (*unexpected_handler) ();
terminate_handler set_terminate(terminate_handler) throw();
void terminate() throw() __attribute__ ((__noreturn__));
unexpected_handler set_unexpected(unexpected_handler) throw();
void unexpected() __attribute__ ((__noreturn__));
bool uncaught_exception() throw() __attribute__ ((__pure__));
}
namespace __gnu_cxx
{
void __verbose_terminate_handler();
}
}
#pragma GCC visibility pop
#pragma GCC visibility push(default)
extern "C++" {
namespace std
{
class bad_alloc : public exception
{
public:
bad_alloc() throw() { }
virtual ~bad_alloc() throw();
virtual const char* what() const throw();
};
struct nothrow_t { };
extern const nothrow_t nothrow;
typedef void (*new_handler)();
new_handler set_new_handler(new_handler) throw();
}
void* operator new(std::size_t) throw(std::bad_alloc)
__attribute__((__externally_visible__));
void* operator new[](std::size_t) throw(std::bad_alloc)
__attribute__((__externally_visible__));
void operator delete(void*) throw()
__attribute__((__externally_visible__));
void operator delete[](void*) throw()
__attribute__((__externally_visible__));
void* operator new(std::size_t, const std::nothrow_t&) throw()
__attribute__((__externally_visible__));
void* operator new[](std::size_t, const std::nothrow_t&) throw()
__attribute__((__externally_visible__));
void operator delete(void*, const std::nothrow_t&) throw()
__attribute__((__externally_visible__));
void operator delete[](void*, const std::nothrow_t&) throw()
__attribute__((__externally_visible__));
inline void* operator new(std::size_t, void* __p) throw()
{ return __p; }
inline void* operator new[](std::size_t, void* __p) throw()
{ return __p; }
inline void operator delete (void*, void*) throw() { }
inline void operator delete[](void*, void*) throw() { }
}
#pragma GCC visibility pop
namespace __gnu_cxx __attribute__ ((__visibility__ ("default")))
{
using std::size_t;
using std::ptrdiff_t;
template<typename _Tp>
class new_allocator
{
public:
typedef size_t size_type;
typedef ptrdiff_t difference_type;
typedef _Tp* pointer;
typedef const _Tp* const_pointer;
typedef _Tp& reference;
typedef const _Tp& const_reference;
typedef _Tp value_type;
template<typename _Tp1>
struct rebind
{ typedef new_allocator<_Tp1> other; };
new_allocator() throw() { }
new_allocator(const new_allocator&) throw() { }
template<typename _Tp1>
new_allocator(const new_allocator<_Tp1>&) throw() { }
~new_allocator() throw() { }
pointer
address(reference __x) const
{ return std::__addressof(__x); }
const_pointer
address(const_reference __x) const
{ return std::__addressof(__x); }
pointer
allocate(size_type __n, const void* = 0)
{
if (__n > this->max_size())
std::__throw_bad_alloc();
return static_cast<_Tp*>(::operator new(__n * sizeof(_Tp)));
}
void
deallocate(pointer __p, size_type)
{ ::operator delete(__p); }
size_type
max_size() const throw()
{ return size_t(-1) / sizeof(_Tp); }
void
construct(pointer __p, const _Tp& __val)
{ ::new((void *)__p) _Tp(__val); }
void
destroy(pointer __p) { __p->~_Tp(); }
};
template<typename _Tp>
inline bool
operator==(const new_allocator<_Tp>&, const new_allocator<_Tp>&)
{ return true; }
template<typename _Tp>
inline bool
operator!=(const new_allocator<_Tp>&, const new_allocator<_Tp>&)
{ return false; }
}
namespace std __attribute__ ((__visibility__ ("default")))
{
template<typename>
class allocator;
template<>
class allocator<void>;
template<typename, typename>
struct uses_allocator;
}
namespace std __attribute__ ((__visibility__ ("default")))
{
template<>
class allocator<void>
{
public:
typedef size_t size_type;
typedef ptrdiff_t difference_type;
typedef void* pointer;
typedef const void* const_pointer;
typedef void value_type;
template<typename _Tp1>
struct rebind
{ typedef allocator<_Tp1> other; };
};
template<typename _Tp>
class allocator: public __gnu_cxx::new_allocator<_Tp>
{
public:
typedef size_t size_type;
typedef ptrdiff_t difference_type;
typedef _Tp* pointer;
typedef const _Tp* const_pointer;
typedef _Tp& reference;
typedef const _Tp& const_reference;
typedef _Tp value_type;
template<typename _Tp1>
struct rebind
{ typedef allocator<_Tp1> other; };
allocator() throw() { }
allocator(const allocator& __a) throw()
: __gnu_cxx::new_allocator<_Tp>(__a) { }
template<typename _Tp1>
allocator(const allocator<_Tp1>&) throw() { }
~allocator() throw() { }
};
template<typename _T1, typename _T2>
inline bool
operator==(const allocator<_T1>&, const allocator<_T2>&)
{ return true; }
template<typename _Tp>
inline bool
operator==(const allocator<_Tp>&, const allocator<_Tp>&)
{ return true; }
template<typename _T1, typename _T2>
inline bool
operator!=(const allocator<_T1>&, const allocator<_T2>&)
{ return false; }
template<typename _Tp>
inline bool
operator!=(const allocator<_Tp>&, const allocator<_Tp>&)
{ return false; }
extern template class allocator<char>;
extern template class allocator<wchar_t>;
template<typename _Alloc, bool = __is_empty(_Alloc)>
struct __alloc_swap
{ static void _S_do_it(_Alloc&, _Alloc&) { } };
template<typename _Alloc>
struct __alloc_swap<_Alloc, false>
{
static void
_S_do_it(_Alloc& __one, _Alloc& __two)
{
if (__one != __two)
swap(__one, __two);
}
};
template<typename _Alloc, bool = __is_empty(_Alloc)>
struct __alloc_neq
{
static bool
_S_do_it(const _Alloc&, const _Alloc&)
{ return false; }
};
template<typename _Alloc>
struct __alloc_neq<_Alloc, false>
{
static bool
_S_do_it(const _Alloc& __one, const _Alloc& __two)
{ return __one != __two; }
};
}
namespace std __attribute__ ((__visibility__ ("default")))
{
template<typename _Arg, typename _Result>
struct unary_function
{
typedef _Arg argument_type;
typedef _Result result_type;
};
template<typename _Arg1, typename _Arg2, typename _Result>
struct binary_function
{
typedef _Arg1 first_argument_type;
typedef _Arg2 second_argument_type;
typedef _Result result_type;
};
template<typename _Tp>
struct plus : public binary_function<_Tp, _Tp, _Tp>
{
_Tp
operator()(const _Tp& __x, const _Tp& __y) const
{ return __x + __y; }
};
template<typename _Tp>
struct minus : public binary_function<_Tp, _Tp, _Tp>
{
_Tp
operator()(const _Tp& __x, const _Tp& __y) const
{ return __x - __y; }
};
template<typename _Tp>
struct multiplies : public binary_function<_Tp, _Tp, _Tp>
{
_Tp
operator()(const _Tp& __x, const _Tp& __y) const
{ return __x * __y; }
};
template<typename _Tp>
struct divides : public binary_function<_Tp, _Tp, _Tp>
{
_Tp
operator()(const _Tp& __x, const _Tp& __y) const
{ return __x / __y; }
};
template<typename _Tp>
struct modulus : public binary_function<_Tp, _Tp, _Tp>
{
_Tp
operator()(const _Tp& __x, const _Tp& __y) const
{ return __x % __y; }
};
template<typename _Tp>
struct negate : public unary_function<_Tp, _Tp>
{
_Tp
operator()(const _Tp& __x) const
{ return -__x; }
};
template<typename _Tp>
struct equal_to : public binary_function<_Tp, _Tp, bool>
{
bool
operator()(const _Tp& __x, const _Tp& __y) const
{ return __x == __y; }
};
template<typename _Tp>
struct not_equal_to : public binary_function<_Tp, _Tp, bool>
{
bool
operator()(const _Tp& __x, const _Tp& __y) const
{ return __x != __y; }
};
template<typename _Tp>
struct greater : public binary_function<_Tp, _Tp, bool>
{
bool
operator()(const _Tp& __x, const _Tp& __y) const
{ return __x > __y; }
};
template<typename _Tp>
struct less : public binary_function<_Tp, _Tp, bool>
{
bool
operator()(const _Tp& __x, const _Tp& __y) const
{ return __x < __y; }
};
template<typename _Tp>
struct greater_equal : public binary_function<_Tp, _Tp, bool>
{
bool
operator()(const _Tp& __x, const _Tp& __y) const
{ return __x >= __y; }
};
template<typename _Tp>
struct less_equal : public binary_function<_Tp, _Tp, bool>
{
bool
operator()(const _Tp& __x, const _Tp& __y) const
{ return __x <= __y; }
};
template<typename _Tp>
struct logical_and : public binary_function<_Tp, _Tp, bool>
{
bool
operator()(const _Tp& __x, const _Tp& __y) const
{ return __x && __y; }
};
template<typename _Tp>
struct logical_or : public binary_function<_Tp, _Tp, bool>
{
bool
operator()(const _Tp& __x, const _Tp& __y) const
{ return __x || __y; }
};
template<typename _Tp>
struct logical_not : public unary_function<_Tp, bool>
{
bool
operator()(const _Tp& __x) const
{ return !__x; }
};
template<typename _Tp>
struct bit_and : public binary_function<_Tp, _Tp, _Tp>
{
_Tp
operator()(const _Tp& __x, const _Tp& __y) const
{ return __x & __y; }
};
template<typename _Tp>
struct bit_or : public binary_function<_Tp, _Tp, _Tp>
{
_Tp
operator()(const _Tp& __x, const _Tp& __y) const
{ return __x | __y; }
};
template<typename _Tp>
struct bit_xor : public binary_function<_Tp, _Tp, _Tp>
{
_Tp
operator()(const _Tp& __x, const _Tp& __y) const
{ return __x ^ __y; }
};
template<typename _Predicate>
class unary_negate
: public unary_function<typename _Predicate::argument_type, bool>
{
protected:
_Predicate _M_pred;
public:
explicit
unary_negate(const _Predicate& __x) : _M_pred(__x) { }
bool
operator()(const typename _Predicate::argument_type& __x) const
{ return !_M_pred(__x); }
};
template<typename _Predicate>
inline unary_negate<_Predicate>
not1(const _Predicate& __pred)
{ return unary_negate<_Predicate>(__pred); }
template<typename _Predicate>
class binary_negate
: public binary_function<typename _Predicate::first_argument_type,
typename _Predicate::second_argument_type, bool>
{
protected:
_Predicate _M_pred;
public:
explicit
binary_negate(const _Predicate& __x) : _M_pred(__x) { }
bool
operator()(const typename _Predicate::first_argument_type& __x,
const typename _Predicate::second_argument_type& __y) const
{ return !_M_pred(__x, __y); }
};
template<typename _Predicate>
inline binary_negate<_Predicate>
not2(const _Predicate& __pred)
{ return binary_negate<_Predicate>(__pred); }
template<typename _Arg, typename _Result>
class pointer_to_unary_function : public unary_function<_Arg, _Result>
{
protected:
_Result (*_M_ptr)(_Arg);
public:
pointer_to_unary_function() { }
explicit
pointer_to_unary_function(_Result (*__x)(_Arg))
: _M_ptr(__x) { }
_Result
operator()(_Arg __x) const
{ return _M_ptr(__x); }
};
template<typename _Arg, typename _Result>
inline pointer_to_unary_function<_Arg, _Result>
ptr_fun(_Result (*__x)(_Arg))
{ return pointer_to_unary_function<_Arg, _Result>(__x); }
template<typename _Arg1, typename _Arg2, typename _Result>
class pointer_to_binary_function
: public binary_function<_Arg1, _Arg2, _Result>
{
protected:
_Result (*_M_ptr)(_Arg1, _Arg2);
public:
pointer_to_binary_function() { }
explicit
pointer_to_binary_function(_Result (*__x)(_Arg1, _Arg2))
: _M_ptr(__x) { }
_Result
operator()(_Arg1 __x, _Arg2 __y) const
{ return _M_ptr(__x, __y); }
};
template<typename _Arg1, typename _Arg2, typename _Result>
inline pointer_to_binary_function<_Arg1, _Arg2, _Result>
ptr_fun(_Result (*__x)(_Arg1, _Arg2))
{ return pointer_to_binary_function<_Arg1, _Arg2, _Result>(__x); }
template<typename _Tp>
struct _Identity
: public unary_function<_Tp,_Tp>
{
_Tp&
operator()(_Tp& __x) const
{ return __x; }
const _Tp&
operator()(const _Tp& __x) const
{ return __x; }
};
template<typename _Pair>
struct _Select1st
: public unary_function<_Pair, typename _Pair::first_type>
{
typename _Pair::first_type&
operator()(_Pair& __x) const
{ return __x.first; }
const typename _Pair::first_type&
operator()(const _Pair& __x) const
{ return __x.first; }
};
template<typename _Pair>
struct _Select2nd
: public unary_function<_Pair, typename _Pair::second_type>
{
typename _Pair::second_type&
operator()(_Pair& __x) const
{ return __x.second; }
const typename _Pair::second_type&
operator()(const _Pair& __x) const
{ return __x.second; }
};
template<typename _Ret, typename _Tp>
class mem_fun_t : public unary_function<_Tp*, _Ret>
{
public:
explicit
mem_fun_t(_Ret (_Tp::*__pf)())
: _M_f(__pf) { }
_Ret
operator()(_Tp* __p) const
{ return (__p->*_M_f)(); }
private:
_Ret (_Tp::*_M_f)();
};
template<typename _Ret, typename _Tp>
class const_mem_fun_t : public unary_function<const _Tp*, _Ret>
{
public:
explicit
const_mem_fun_t(_Ret (_Tp::*__pf)() const)
: _M_f(__pf) { }
_Ret
operator()(const _Tp* __p) const
{ return (__p->*_M_f)(); }
private:
_Ret (_Tp::*_M_f)() const;
};
template<typename _Ret, typename _Tp>
class mem_fun_ref_t : public unary_function<_Tp, _Ret>
{
public:
explicit
mem_fun_ref_t(_Ret (_Tp::*__pf)())
: _M_f(__pf) { }
_Ret
operator()(_Tp& __r) const
{ return (__r.*_M_f)(); }
private:
_Ret (_Tp::*_M_f)();
};
template<typename _Ret, typename _Tp>
class const_mem_fun_ref_t : public unary_function<_Tp, _Ret>
{
public:
explicit
const_mem_fun_ref_t(_Ret (_Tp::*__pf)() const)
: _M_f(__pf) { }
_Ret
operator()(const _Tp& __r) const
{ return (__r.*_M_f)(); }
private:
_Ret (_Tp::*_M_f)() const;
};
template<typename _Ret, typename _Tp, typename _Arg>
class mem_fun1_t : public binary_function<_Tp*, _Arg, _Ret>
{
public:
explicit
mem_fun1_t(_Ret (_Tp::*__pf)(_Arg))
: _M_f(__pf) { }
_Ret
operator()(_Tp* __p, _Arg __x) const
{ return (__p->*_M_f)(__x); }
private:
_Ret (_Tp::*_M_f)(_Arg);
};
template<typename _Ret, typename _Tp, typename _Arg>
class const_mem_fun1_t : public binary_function<const _Tp*, _Arg, _Ret>
{
public:
explicit
const_mem_fun1_t(_Ret (_Tp::*__pf)(_Arg) const)
: _M_f(__pf) { }
_Ret
operator()(const _Tp* __p, _Arg __x) const
{ return (__p->*_M_f)(__x); }
private:
_Ret (_Tp::*_M_f)(_Arg) const;
};
template<typename _Ret, typename _Tp, typename _Arg>
class mem_fun1_ref_t : public binary_function<_Tp, _Arg, _Ret>
{
public:
explicit
mem_fun1_ref_t(_Ret (_Tp::*__pf)(_Arg))
: _M_f(__pf) { }
_Ret
operator()(_Tp& __r, _Arg __x) const
{ return (__r.*_M_f)(__x); }
private:
_Ret (_Tp::*_M_f)(_Arg);
};
template<typename _Ret, typename _Tp, typename _Arg>
class const_mem_fun1_ref_t : public binary_function<_Tp, _Arg, _Ret>
{
public:
explicit
const_mem_fun1_ref_t(_Ret (_Tp::*__pf)(_Arg) const)
: _M_f(__pf) { }
_Ret
operator()(const _Tp& __r, _Arg __x) const
{ return (__r.*_M_f)(__x); }
private:
_Ret (_Tp::*_M_f)(_Arg) const;
};
template<typename _Ret, typename _Tp>
inline mem_fun_t<_Ret, _Tp>
mem_fun(_Ret (_Tp::*__f)())
{ return mem_fun_t<_Ret, _Tp>(__f); }
template<typename _Ret, typename _Tp>
inline const_mem_fun_t<_Ret, _Tp>
mem_fun(_Ret (_Tp::*__f)() const)
{ return const_mem_fun_t<_Ret, _Tp>(__f); }
template<typename _Ret, typename _Tp>
inline mem_fun_ref_t<_Ret, _Tp>
mem_fun_ref(_Ret (_Tp::*__f)())
{ return mem_fun_ref_t<_Ret, _Tp>(__f); }
template<typename _Ret, typename _Tp>
inline const_mem_fun_ref_t<_Ret, _Tp>
mem_fun_ref(_Ret (_Tp::*__f)() const)
{ return const_mem_fun_ref_t<_Ret, _Tp>(__f); }
template<typename _Ret, typename _Tp, typename _Arg>
inline mem_fun1_t<_Ret, _Tp, _Arg>
mem_fun(_Ret (_Tp::*__f)(_Arg))
{ return mem_fun1_t<_Ret, _Tp, _Arg>(__f); }
template<typename _Ret, typename _Tp, typename _Arg>
inline const_mem_fun1_t<_Ret, _Tp, _Arg>
mem_fun(_Ret (_Tp::*__f)(_Arg) const)
{ return const_mem_fun1_t<_Ret, _Tp, _Arg>(__f); }
template<typename _Ret, typename _Tp, typename _Arg>
inline mem_fun1_ref_t<_Ret, _Tp, _Arg>
mem_fun_ref(_Ret (_Tp::*__f)(_Arg))
{ return mem_fun1_ref_t<_Ret, _Tp, _Arg>(__f); }
template<typename _Ret, typename _Tp, typename _Arg>
inline const_mem_fun1_ref_t<_Ret, _Tp, _Arg>
mem_fun_ref(_Ret (_Tp::*__f)(_Arg) const)
{ return const_mem_fun1_ref_t<_Ret, _Tp, _Arg>(__f); }
}
namespace std __attribute__ ((__visibility__ ("default")))
{
template<typename _Operation>
class binder1st
: public unary_function<typename _Operation::second_argument_type,
typename _Operation::result_type>
{
protected:
_Operation op;
typename _Operation::first_argument_type value;
public:
binder1st(const _Operation& __x,
const typename _Operation::first_argument_type& __y)
: op(__x), value(__y) { }
typename _Operation::result_type
operator()(const typename _Operation::second_argument_type& __x) const
{ return op(value, __x); }
typename _Operation::result_type
operator()(typename _Operation::second_argument_type& __x) const
{ return op(value, __x); }
} ;
template<typename _Operation, typename _Tp>
inline binder1st<_Operation>
bind1st(const _Operation& __fn, const _Tp& __x)
{
typedef typename _Operation::first_argument_type _Arg1_type;
return binder1st<_Operation>(__fn, _Arg1_type(__x));
}
template<typename _Operation>
class binder2nd
: public unary_function<typename _Operation::first_argument_type,
typename _Operation::result_type>
{
protected:
_Operation op;
typename _Operation::second_argument_type value;
public:
binder2nd(const _Operation& __x,
const typename _Operation::second_argument_type& __y)
: op(__x), value(__y) { }
typename _Operation::result_type
operator()(const typename _Operation::first_argument_type& __x) const
{ return op(__x, value); }
typename _Operation::result_type
operator()(typename _Operation::first_argument_type& __x) const
{ return op(__x, value); }
} ;
template<typename _Operation, typename _Tp>
inline binder2nd<_Operation>
bind2nd(const _Operation& __fn, const _Tp& __x)
{
typedef typename _Operation::second_argument_type _Arg2_type;
return binder2nd<_Operation>(__fn, _Arg2_type(__x));
}
}
namespace std __attribute__ ((__visibility__ ("default")))
{
enum _Rb_tree_color { _S_red = false, _S_black = true };
struct _Rb_tree_node_base
{
typedef _Rb_tree_node_base* _Base_ptr;
typedef const _Rb_tree_node_base* _Const_Base_ptr;
_Rb_tree_color _M_color;
_Base_ptr _M_parent;
_Base_ptr _M_left;
_Base_ptr _M_right;
static _Base_ptr
_S_minimum(_Base_ptr __x)
{
while (__x->_M_left != 0) __x = __x->_M_left;
return __x;
}
static _Const_Base_ptr
_S_minimum(_Const_Base_ptr __x)
{
while (__x->_M_left != 0) __x = __x->_M_left;
return __x;
}
static _Base_ptr
_S_maximum(_Base_ptr __x)
{
while (__x->_M_right != 0) __x = __x->_M_right;
return __x;
}
static _Const_Base_ptr
_S_maximum(_Const_Base_ptr __x)
{
while (__x->_M_right != 0) __x = __x->_M_right;
return __x;
}
};
template<typename _Val>
struct _Rb_tree_node : public _Rb_tree_node_base
{
typedef _Rb_tree_node<_Val>* _Link_type;
_Val _M_value_field;
};
__attribute__ ((__pure__)) _Rb_tree_node_base*
_Rb_tree_increment(_Rb_tree_node_base* __x) throw ();
__attribute__ ((__pure__)) const _Rb_tree_node_base*
_Rb_tree_increment(const _Rb_tree_node_base* __x) throw ();
__attribute__ ((__pure__)) _Rb_tree_node_base*
_Rb_tree_decrement(_Rb_tree_node_base* __x) throw ();
__attribute__ ((__pure__)) const _Rb_tree_node_base*
_Rb_tree_decrement(const _Rb_tree_node_base* __x) throw ();
template<typename _Tp>
struct _Rb_tree_iterator
{
typedef _Tp value_type;
typedef _Tp& reference;
typedef _Tp* pointer;
typedef bidirectional_iterator_tag iterator_category;
typedef ptrdiff_t difference_type;
typedef _Rb_tree_iterator<_Tp> _Self;
typedef _Rb_tree_node_base::_Base_ptr _Base_ptr;
typedef _Rb_tree_node<_Tp>* _Link_type;
_Rb_tree_iterator()
: _M_node() { }
explicit
_Rb_tree_iterator(_Link_type __x)
: _M_node(__x) { }
reference
operator*() const
{ return static_cast<_Link_type>(_M_node)->_M_value_field; }
pointer
operator->() const
{ return std::__addressof(static_cast<_Link_type>
(_M_node)->_M_value_field); }
_Self&
operator++()
{
_M_node = _Rb_tree_increment(_M_node);
return *this;
}
_Self
operator++(int)
{
_Self __tmp = *this;
_M_node = _Rb_tree_increment(_M_node);
return __tmp;
}
_Self&
operator--()
{
_M_node = _Rb_tree_decrement(_M_node);
return *this;
}
_Self
operator--(int)
{
_Self __tmp = *this;
_M_node = _Rb_tree_decrement(_M_node);
return __tmp;
}
bool
operator==(const _Self& __x) const
{ return _M_node == __x._M_node; }
bool
operator!=(const _Self& __x) const
{ return _M_node != __x._M_node; }
_Base_ptr _M_node;
};
template<typename _Tp>
struct _Rb_tree_const_iterator
{
typedef _Tp value_type;
typedef const _Tp& reference;
typedef const _Tp* pointer;
typedef _Rb_tree_iterator<_Tp> iterator;
typedef bidirectional_iterator_tag iterator_category;
typedef ptrdiff_t difference_type;
typedef _Rb_tree_const_iterator<_Tp> _Self;
typedef _Rb_tree_node_base::_Const_Base_ptr _Base_ptr;
typedef const _Rb_tree_node<_Tp>* _Link_type;
_Rb_tree_const_iterator()
: _M_node() { }
explicit
_Rb_tree_const_iterator(_Link_type __x)
: _M_node(__x) { }
_Rb_tree_const_iterator(const iterator& __it)
: _M_node(__it._M_node) { }
iterator
_M_const_cast() const
{ return iterator(static_cast<typename iterator::_Link_type>
(const_cast<typename iterator::_Base_ptr>(_M_node))); }
reference
operator*() const
{ return static_cast<_Link_type>(_M_node)->_M_value_field; }
pointer
operator->() const
{ return std::__addressof(static_cast<_Link_type>
(_M_node)->_M_value_field); }
_Self&
operator++()
{
_M_node = _Rb_tree_increment(_M_node);
return *this;
}
_Self
operator++(int)
{
_Self __tmp = *this;
_M_node = _Rb_tree_increment(_M_node);
return __tmp;
}
_Self&
operator--()
{
_M_node = _Rb_tree_decrement(_M_node);
return *this;
}
_Self
operator--(int)
{
_Self __tmp = *this;
_M_node = _Rb_tree_decrement(_M_node);
return __tmp;
}
bool
operator==(const _Self& __x) const
{ return _M_node == __x._M_node; }
bool
operator!=(const _Self& __x) const
{ return _M_node != __x._M_node; }
_Base_ptr _M_node;
};
template<typename _Val>
inline bool
operator==(const _Rb_tree_iterator<_Val>& __x,
const _Rb_tree_const_iterator<_Val>& __y)
{ return __x._M_node == __y._M_node; }
template<typename _Val>
inline bool
operator!=(const _Rb_tree_iterator<_Val>& __x,
const _Rb_tree_const_iterator<_Val>& __y)
{ return __x._M_node != __y._M_node; }
void
_Rb_tree_insert_and_rebalance(const bool __insert_left,
_Rb_tree_node_base* __x,
_Rb_tree_node_base* __p,
_Rb_tree_node_base& __header) throw ();
_Rb_tree_node_base*
_Rb_tree_rebalance_for_erase(_Rb_tree_node_base* const __z,
_Rb_tree_node_base& __header) throw ();
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc = allocator<_Val> >
class _Rb_tree
{
typedef typename _Alloc::template rebind<_Rb_tree_node<_Val> >::other
_Node_allocator;
protected:
typedef _Rb_tree_node_base* _Base_ptr;
typedef const _Rb_tree_node_base* _Const_Base_ptr;
public:
typedef _Key key_type;
typedef _Val value_type;
typedef value_type* pointer;
typedef const value_type* const_pointer;
typedef value_type& reference;
typedef const value_type& const_reference;
typedef _Rb_tree_node<_Val>* _Link_type;
typedef const _Rb_tree_node<_Val>* _Const_Link_type;
typedef size_t size_type;
typedef ptrdiff_t difference_type;
typedef _Alloc allocator_type;
_Node_allocator&
_M_get_Node_allocator()
{ return *static_cast<_Node_allocator*>(&this->_M_impl); }
const _Node_allocator&
_M_get_Node_allocator() const
{ return *static_cast<const _Node_allocator*>(&this->_M_impl); }
allocator_type
get_allocator() const
{ return allocator_type(_M_get_Node_allocator()); }
protected:
_Link_type
_M_get_node()
{ return _M_impl._Node_allocator::allocate(1); }
void
_M_put_node(_Link_type __p)
{ _M_impl._Node_allocator::deallocate(__p, 1); }
_Link_type
_M_create_node(const value_type& __x)
{
_Link_type __tmp = _M_get_node();
try
{ get_allocator().construct
(std::__addressof(__tmp->_M_value_field), __x); }
catch(...)
{
_M_put_node(__tmp);
throw;
}
return __tmp;
}
void
_M_destroy_node(_Link_type __p)
{
get_allocator().destroy(std::__addressof(__p->_M_value_field));
_M_put_node(__p);
}
_Link_type
_M_clone_node(_Const_Link_type __x)
{
_Link_type __tmp = _M_create_node(__x->_M_value_field);
__tmp->_M_color = __x->_M_color;
__tmp->_M_left = 0;
__tmp->_M_right = 0;
return __tmp;
}
protected:
template<typename _Key_compare,
bool _Is_pod_comparator = __is_pod(_Key_compare)>
struct _Rb_tree_impl : public _Node_allocator
{
_Key_compare _M_key_compare;
_Rb_tree_node_base _M_header;
size_type _M_node_count;
_Rb_tree_impl()
: _Node_allocator(), _M_key_compare(), _M_header(),
_M_node_count(0)
{ _M_initialize(); }
_Rb_tree_impl(const _Key_compare& __comp, const _Node_allocator& __a)
: _Node_allocator(__a), _M_key_compare(__comp), _M_header(),
_M_node_count(0)
{ _M_initialize(); }
private:
void
_M_initialize()
{
this->_M_header._M_color = _S_red;
this->_M_header._M_parent = 0;
this->_M_header._M_left = &this->_M_header;
this->_M_header._M_right = &this->_M_header;
}
};
_Rb_tree_impl<_Compare> _M_impl;
protected:
_Base_ptr&
_M_root()
{ return this->_M_impl._M_header._M_parent; }
_Const_Base_ptr
_M_root() const
{ return this->_M_impl._M_header._M_parent; }
_Base_ptr&
_M_leftmost()
{ return this->_M_impl._M_header._M_left; }
_Const_Base_ptr
_M_leftmost() const
{ return this->_M_impl._M_header._M_left; }
_Base_ptr&
_M_rightmost()
{ return this->_M_impl._M_header._M_right; }
_Const_Base_ptr
_M_rightmost() const
{ return this->_M_impl._M_header._M_right; }
_Link_type
_M_begin()
{ return static_cast<_Link_type>(this->_M_impl._M_header._M_parent); }
_Const_Link_type
_M_begin() const
{
return static_cast<_Const_Link_type>
(this->_M_impl._M_header._M_parent);
}
_Link_type
_M_end()
{ return static_cast<_Link_type>(&this->_M_impl._M_header); }
_Const_Link_type
_M_end() const
{ return static_cast<_Const_Link_type>(&this->_M_impl._M_header); }
static const_reference
_S_value(_Const_Link_type __x)
{ return __x->_M_value_field; }
static const _Key&
_S_key(_Const_Link_type __x)
{ return _KeyOfValue()(_S_value(__x)); }
static _Link_type
_S_left(_Base_ptr __x)
{ return static_cast<_Link_type>(__x->_M_left); }
static _Const_Link_type
_S_left(_Const_Base_ptr __x)
{ return static_cast<_Const_Link_type>(__x->_M_left); }
static _Link_type
_S_right(_Base_ptr __x)
{ return static_cast<_Link_type>(__x->_M_right); }
static _Const_Link_type
_S_right(_Const_Base_ptr __x)
{ return static_cast<_Const_Link_type>(__x->_M_right); }
static const_reference
_S_value(_Const_Base_ptr __x)
{ return static_cast<_Const_Link_type>(__x)->_M_value_field; }
static const _Key&
_S_key(_Const_Base_ptr __x)
{ return _KeyOfValue()(_S_value(__x)); }
static _Base_ptr
_S_minimum(_Base_ptr __x)
{ return _Rb_tree_node_base::_S_minimum(__x); }
static _Const_Base_ptr
_S_minimum(_Const_Base_ptr __x)
{ return _Rb_tree_node_base::_S_minimum(__x); }
static _Base_ptr
_S_maximum(_Base_ptr __x)
{ return _Rb_tree_node_base::_S_maximum(__x); }
static _Const_Base_ptr
_S_maximum(_Const_Base_ptr __x)
{ return _Rb_tree_node_base::_S_maximum(__x); }
public:
typedef _Rb_tree_iterator<value_type> iterator;
typedef _Rb_tree_const_iterator<value_type> const_iterator;
typedef std::reverse_iterator<iterator> reverse_iterator;
typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
private:
pair<_Base_ptr, _Base_ptr>
_M_get_insert_unique_pos(const key_type& __k);
pair<_Base_ptr, _Base_ptr>
_M_get_insert_equal_pos(const key_type& __k);
pair<_Base_ptr, _Base_ptr>
_M_get_insert_hint_unique_pos(const_iterator __pos,
const key_type& __k);
pair<_Base_ptr, _Base_ptr>
_M_get_insert_hint_equal_pos(const_iterator __pos,
const key_type& __k);
iterator
_M_insert_(_Base_ptr __x, _Base_ptr __y,
const value_type& __v);
iterator
_M_insert_lower(_Base_ptr __y, const value_type& __v);
iterator
_M_insert_equal_lower(const value_type& __x);
_Link_type
_M_copy(_Const_Link_type __x, _Link_type __p);
void
_M_erase(_Link_type __x);
iterator
_M_lower_bound(_Link_type __x, _Link_type __y,
const _Key& __k);
const_iterator
_M_lower_bound(_Const_Link_type __x, _Const_Link_type __y,
const _Key& __k) const;
iterator
_M_upper_bound(_Link_type __x, _Link_type __y,
const _Key& __k);
const_iterator
_M_upper_bound(_Const_Link_type __x, _Const_Link_type __y,
const _Key& __k) const;
public:
_Rb_tree() { }
_Rb_tree(const _Compare& __comp,
const allocator_type& __a = allocator_type())
: _M_impl(__comp, _Node_allocator(__a)) { }
_Rb_tree(const _Rb_tree& __x)
: _M_impl(__x._M_impl._M_key_compare, __x._M_get_Node_allocator())
{
if (__x._M_root() != 0)
{
_M_root() = _M_copy(__x._M_begin(), _M_end());
_M_leftmost() = _S_minimum(_M_root());
_M_rightmost() = _S_maximum(_M_root());
_M_impl._M_node_count = __x._M_impl._M_node_count;
}
}
~_Rb_tree()
{ _M_erase(_M_begin()); }
_Rb_tree&
operator=(const _Rb_tree& __x);
_Compare
key_comp() const
{ return _M_impl._M_key_compare; }
iterator
begin()
{
return iterator(static_cast<_Link_type>
(this->_M_impl._M_header._M_left));
}
const_iterator
begin() const
{
return const_iterator(static_cast<_Const_Link_type>
(this->_M_impl._M_header._M_left));
}
iterator
end()
{ return iterator(static_cast<_Link_type>(&this->_M_impl._M_header)); }
const_iterator
end() const
{
return const_iterator(static_cast<_Const_Link_type>
(&this->_M_impl._M_header));
}
reverse_iterator
rbegin()
{ return reverse_iterator(end()); }
const_reverse_iterator
rbegin() const
{ return const_reverse_iterator(end()); }
reverse_iterator
rend()
{ return reverse_iterator(begin()); }
const_reverse_iterator
rend() const
{ return const_reverse_iterator(begin()); }
bool
empty() const
{ return _M_impl._M_node_count == 0; }
size_type
size() const
{ return _M_impl._M_node_count; }
size_type
max_size() const
{ return _M_get_Node_allocator().max_size(); }
void
swap(_Rb_tree& __t);
pair<iterator, bool>
_M_insert_unique(const value_type& __x);
iterator
_M_insert_equal(const value_type& __x);
iterator
_M_insert_unique_(const_iterator __position, const value_type& __x);
iterator
_M_insert_equal_(const_iterator __position, const value_type& __x);
template<typename _InputIterator>
void
_M_insert_unique(_InputIterator __first, _InputIterator __last);
template<typename _InputIterator>
void
_M_insert_equal(_InputIterator __first, _InputIterator __last);
private:
void
_M_erase_aux(const_iterator __position);
void
_M_erase_aux(const_iterator __first, const_iterator __last);
public:
void
erase(iterator __position)
{ _M_erase_aux(__position); }
void
erase(const_iterator __position)
{ _M_erase_aux(__position); }
size_type
erase(const key_type& __x);
void
erase(iterator __first, iterator __last)
{ _M_erase_aux(__first, __last); }
void
erase(const_iterator __first, const_iterator __last)
{ _M_erase_aux(__first, __last); }
void
erase(const key_type* __first, const key_type* __last);
void
clear()
{
_M_erase(_M_begin());
_M_leftmost() = _M_end();
_M_root() = 0;
_M_rightmost() = _M_end();
_M_impl._M_node_count = 0;
}
iterator
find(const key_type& __k);
const_iterator
find(const key_type& __k) const;
size_type
count(const key_type& __k) const;
iterator
lower_bound(const key_type& __k)
{ return _M_lower_bound(_M_begin(), _M_end(), __k); }
const_iterator
lower_bound(const key_type& __k) const
{ return _M_lower_bound(_M_begin(), _M_end(), __k); }
iterator
upper_bound(const key_type& __k)
{ return _M_upper_bound(_M_begin(), _M_end(), __k); }
const_iterator
upper_bound(const key_type& __k) const
{ return _M_upper_bound(_M_begin(), _M_end(), __k); }
pair<iterator, iterator>
equal_range(const key_type& __k);
pair<const_iterator, const_iterator>
equal_range(const key_type& __k) const;
bool
__rb_verify() const;
};
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
inline bool
operator==(const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>& __x,
const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>& __y)
{
return __x.size() == __y.size()
&& std::equal(__x.begin(), __x.end(), __y.begin());
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
inline bool
operator<(const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>& __x,
const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>& __y)
{
return std::lexicographical_compare(__x.begin(), __x.end(),
__y.begin(), __y.end());
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
inline bool
operator!=(const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>& __x,
const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>& __y)
{ return !(__x == __y); }
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
inline bool
operator>(const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>& __x,
const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>& __y)
{ return __y < __x; }
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
inline bool
operator<=(const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>& __x,
const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>& __y)
{ return !(__y < __x); }
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
inline bool
operator>=(const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>& __x,
const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>& __y)
{ return !(__x < __y); }
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
inline void
swap(_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>& __x,
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>& __y)
{ __x.swap(__y); }
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>&
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
operator=(const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>& __x)
{
if (this != &__x)
{
clear();
_M_impl._M_key_compare = __x._M_impl._M_key_compare;
if (__x._M_root() != 0)
{
_M_root() = _M_copy(__x._M_begin(), _M_end());
_M_leftmost() = _S_minimum(_M_root());
_M_rightmost() = _S_maximum(_M_root());
_M_impl._M_node_count = __x._M_impl._M_node_count;
}
}
return *this;
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_insert_(_Base_ptr __x, _Base_ptr __p, const _Val& __v)
{
bool __insert_left = (__x != 0 || __p == _M_end()
|| _M_impl._M_key_compare(_KeyOfValue()(__v),
_S_key(__p)));
_Link_type __z = _M_create_node((__v));
_Rb_tree_insert_and_rebalance(__insert_left, __z, __p,
this->_M_impl._M_header);
++_M_impl._M_node_count;
return iterator(__z);
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_insert_lower(_Base_ptr __p, const _Val& __v)
{
bool __insert_left = (__p == _M_end()
|| !_M_impl._M_key_compare(_S_key(__p),
_KeyOfValue()(__v)));
_Link_type __z = _M_create_node((__v));
_Rb_tree_insert_and_rebalance(__insert_left, __z, __p,
this->_M_impl._M_header);
++_M_impl._M_node_count;
return iterator(__z);
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_insert_equal_lower(const _Val& __v)
{
_Link_type __x = _M_begin();
_Link_type __y = _M_end();
while (__x != 0)
{
__y = __x;
__x = !_M_impl._M_key_compare(_S_key(__x), _KeyOfValue()(__v)) ?
_S_left(__x) : _S_right(__x);
}
return _M_insert_lower(__y, (__v));
}
template<typename _Key, typename _Val, typename _KoV,
typename _Compare, typename _Alloc>
typename _Rb_tree<_Key, _Val, _KoV, _Compare, _Alloc>::_Link_type
_Rb_tree<_Key, _Val, _KoV, _Compare, _Alloc>::
_M_copy(_Const_Link_type __x, _Link_type __p)
{
_Link_type __top = _M_clone_node(__x);
__top->_M_parent = __p;
try
{
if (__x->_M_right)
__top->_M_right = _M_copy(_S_right(__x), __top);
__p = __top;
__x = _S_left(__x);
while (__x != 0)
{
_Link_type __y = _M_clone_node(__x);
__p->_M_left = __y;
__y->_M_parent = __p;
if (__x->_M_right)
__y->_M_right = _M_copy(_S_right(__x), __y);
__p = __y;
__x = _S_left(__x);
}
}
catch(...)
{
_M_erase(__top);
throw;
}
return __top;
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
void
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_erase(_Link_type __x)
{
while (__x != 0)
{
_M_erase(_S_right(__x));
_Link_type __y = _S_left(__x);
_M_destroy_node(__x);
__x = __y;
}
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::iterator
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_lower_bound(_Link_type __x, _Link_type __y,
const _Key& __k)
{
while (__x != 0)
if (!_M_impl._M_key_compare(_S_key(__x), __k))
__y = __x, __x = _S_left(__x);
else
__x = _S_right(__x);
return iterator(__y);
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::const_iterator
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_lower_bound(_Const_Link_type __x, _Const_Link_type __y,
const _Key& __k) const
{
while (__x != 0)
if (!_M_impl._M_key_compare(_S_key(__x), __k))
__y = __x, __x = _S_left(__x);
else
__x = _S_right(__x);
return const_iterator(__y);
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::iterator
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_upper_bound(_Link_type __x, _Link_type __y,
const _Key& __k)
{
while (__x != 0)
if (_M_impl._M_key_compare(__k, _S_key(__x)))
__y = __x, __x = _S_left(__x);
else
__x = _S_right(__x);
return iterator(__y);
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::const_iterator
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_upper_bound(_Const_Link_type __x, _Const_Link_type __y,
const _Key& __k) const
{
while (__x != 0)
if (_M_impl._M_key_compare(__k, _S_key(__x)))
__y = __x, __x = _S_left(__x);
else
__x = _S_right(__x);
return const_iterator(__y);
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
pair<typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::iterator,
typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::iterator>
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
equal_range(const _Key& __k)
{
_Link_type __x = _M_begin();
_Link_type __y = _M_end();
while (__x != 0)
{
if (_M_impl._M_key_compare(_S_key(__x), __k))
__x = _S_right(__x);
else if (_M_impl._M_key_compare(__k, _S_key(__x)))
__y = __x, __x = _S_left(__x);
else
{
_Link_type __xu(__x), __yu(__y);
__y = __x, __x = _S_left(__x);
__xu = _S_right(__xu);
return pair<iterator,
iterator>(_M_lower_bound(__x, __y, __k),
_M_upper_bound(__xu, __yu, __k));
}
}
return pair<iterator, iterator>(iterator(__y),
iterator(__y));
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
pair<typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::const_iterator,
typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::const_iterator>
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
equal_range(const _Key& __k) const
{
_Const_Link_type __x = _M_begin();
_Const_Link_type __y = _M_end();
while (__x != 0)
{
if (_M_impl._M_key_compare(_S_key(__x), __k))
__x = _S_right(__x);
else if (_M_impl._M_key_compare(__k, _S_key(__x)))
__y = __x, __x = _S_left(__x);
else
{
_Const_Link_type __xu(__x), __yu(__y);
__y = __x, __x = _S_left(__x);
__xu = _S_right(__xu);
return pair<const_iterator,
const_iterator>(_M_lower_bound(__x, __y, __k),
_M_upper_bound(__xu, __yu, __k));
}
}
return pair<const_iterator, const_iterator>(const_iterator(__y),
const_iterator(__y));
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
void
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
swap(_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>& __t)
{
if (_M_root() == 0)
{
if (__t._M_root() != 0)
{
_M_root() = __t._M_root();
_M_leftmost() = __t._M_leftmost();
_M_rightmost() = __t._M_rightmost();
_M_root()->_M_parent = _M_end();
__t._M_root() = 0;
__t._M_leftmost() = __t._M_end();
__t._M_rightmost() = __t._M_end();
}
}
else if (__t._M_root() == 0)
{
__t._M_root() = _M_root();
__t._M_leftmost() = _M_leftmost();
__t._M_rightmost() = _M_rightmost();
__t._M_root()->_M_parent = __t._M_end();
_M_root() = 0;
_M_leftmost() = _M_end();
_M_rightmost() = _M_end();
}
else
{
std::swap(_M_root(),__t._M_root());
std::swap(_M_leftmost(),__t._M_leftmost());
std::swap(_M_rightmost(),__t._M_rightmost());
_M_root()->_M_parent = _M_end();
__t._M_root()->_M_parent = __t._M_end();
}
std::swap(this->_M_impl._M_node_count, __t._M_impl._M_node_count);
std::swap(this->_M_impl._M_key_compare, __t._M_impl._M_key_compare);
std::__alloc_swap<_Node_allocator>::
_S_do_it(_M_get_Node_allocator(), __t._M_get_Node_allocator());
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
pair<typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::_Base_ptr,
typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::_Base_ptr>
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_get_insert_unique_pos(const key_type& __k)
{
typedef pair<_Base_ptr, _Base_ptr> _Res;
_Link_type __x = _M_begin();
_Link_type __y = _M_end();
bool __comp = true;
while (__x != 0)
{
__y = __x;
__comp = _M_impl._M_key_compare(__k, _S_key(__x));
__x = __comp ? _S_left(__x) : _S_right(__x);
}
iterator __j = iterator(__y);
if (__comp)
{
if (__j == begin())
return _Res(__x, __y);
else
--__j;
}
if (_M_impl._M_key_compare(_S_key(__j._M_node), __k))
return _Res(__x, __y);
return _Res(__j._M_node, 0);
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
pair<typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::_Base_ptr,
typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::_Base_ptr>
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_get_insert_equal_pos(const key_type& __k)
{
typedef pair<_Base_ptr, _Base_ptr> _Res;
_Link_type __x = _M_begin();
_Link_type __y = _M_end();
while (__x != 0)
{
__y = __x;
__x = _M_impl._M_key_compare(__k, _S_key(__x)) ?
_S_left(__x) : _S_right(__x);
}
return _Res(__x, __y);
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
pair<typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::iterator, bool>
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_insert_unique(const _Val& __v)
{
typedef pair<iterator, bool> _Res;
pair<_Base_ptr, _Base_ptr> __res
= _M_get_insert_unique_pos(_KeyOfValue()(__v));
if (__res.second)
return _Res(_M_insert_(__res.first, __res.second,
(__v)),
true);
return _Res(iterator(static_cast<_Link_type>(__res.first)), false);
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_insert_equal(const _Val& __v)
{
pair<_Base_ptr, _Base_ptr> __res
= _M_get_insert_equal_pos(_KeyOfValue()(__v));
return _M_insert_(__res.first, __res.second, (__v));
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
pair<typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::_Base_ptr,
typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::_Base_ptr>
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_get_insert_hint_unique_pos(const_iterator __position,
const key_type& __k)
{
iterator __pos = __position._M_const_cast();
typedef pair<_Base_ptr, _Base_ptr> _Res;
if (__pos._M_node == _M_end())
{
if (size() > 0
&& _M_impl._M_key_compare(_S_key(_M_rightmost()), __k))
return _Res(0, _M_rightmost());
else
return _M_get_insert_unique_pos(__k);
}
else if (_M_impl._M_key_compare(__k, _S_key(__pos._M_node)))
{
iterator __before = __pos;
if (__pos._M_node == _M_leftmost())
return _Res(_M_leftmost(), _M_leftmost());
else if (_M_impl._M_key_compare(_S_key((--__before)._M_node), __k))
{
if (_S_right(__before._M_node) == 0)
return _Res(0, __before._M_node);
else
return _Res(__pos._M_node, __pos._M_node);
}
else
return _M_get_insert_unique_pos(__k);
}
else if (_M_impl._M_key_compare(_S_key(__pos._M_node), __k))
{
iterator __after = __pos;
if (__pos._M_node == _M_rightmost())
return _Res(0, _M_rightmost());
else if (_M_impl._M_key_compare(__k, _S_key((++__after)._M_node)))
{
if (_S_right(__pos._M_node) == 0)
return _Res(0, __pos._M_node);
else
return _Res(__after._M_node, __after._M_node);
}
else
return _M_get_insert_unique_pos(__k);
}
else
return _Res(__pos._M_node, 0);
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_insert_unique_(const_iterator __position, const _Val& __v)
{
pair<_Base_ptr, _Base_ptr> __res
= _M_get_insert_hint_unique_pos(__position, _KeyOfValue()(__v));
if (__res.second)
return _M_insert_(__res.first, __res.second,
(__v));
return iterator(static_cast<_Link_type>(__res.first));
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
pair<typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::_Base_ptr,
typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::_Base_ptr>
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_get_insert_hint_equal_pos(const_iterator __position, const key_type& __k)
{
iterator __pos = __position._M_const_cast();
typedef pair<_Base_ptr, _Base_ptr> _Res;
if (__pos._M_node == _M_end())
{
if (size() > 0
&& !_M_impl._M_key_compare(__k, _S_key(_M_rightmost())))
return _Res(0, _M_rightmost());
else
return _M_get_insert_equal_pos(__k);
}
else if (!_M_impl._M_key_compare(_S_key(__pos._M_node), __k))
{
iterator __before = __pos;
if (__pos._M_node == _M_leftmost())
return _Res(_M_leftmost(), _M_leftmost());
else if (!_M_impl._M_key_compare(__k, _S_key((--__before)._M_node)))
{
if (_S_right(__before._M_node) == 0)
return _Res(0, __before._M_node);
else
return _Res(__pos._M_node, __pos._M_node);
}
else
return _M_get_insert_equal_pos(__k);
}
else
{
iterator __after = __pos;
if (__pos._M_node == _M_rightmost())
return _Res(0, _M_rightmost());
else if (!_M_impl._M_key_compare(_S_key((++__after)._M_node), __k))
{
if (_S_right(__pos._M_node) == 0)
return _Res(0, __pos._M_node);
else
return _Res(__after._M_node, __after._M_node);
}
else
return _Res(0, 0);
}
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_insert_equal_(const_iterator __position, const _Val& __v)
{
pair<_Base_ptr, _Base_ptr> __res
= _M_get_insert_hint_equal_pos(__position, _KeyOfValue()(__v));
if (__res.second)
return _M_insert_(__res.first, __res.second,
(__v));
return _M_insert_equal_lower((__v));
}
template<typename _Key, typename _Val, typename _KoV,
typename _Cmp, typename _Alloc>
template<class _II>
void
_Rb_tree<_Key, _Val, _KoV, _Cmp, _Alloc>::
_M_insert_unique(_II __first, _II __last)
{
for (; __first != __last; ++__first)
_M_insert_unique_(end(), *__first);
}
template<typename _Key, typename _Val, typename _KoV,
typename _Cmp, typename _Alloc>
template<class _II>
void
_Rb_tree<_Key, _Val, _KoV, _Cmp, _Alloc>::
_M_insert_equal(_II __first, _II __last)
{
for (; __first != __last; ++__first)
_M_insert_equal_(end(), *__first);
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
void
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_erase_aux(const_iterator __position)
{
_Link_type __y =
static_cast<_Link_type>(_Rb_tree_rebalance_for_erase
(const_cast<_Base_ptr>(__position._M_node),
this->_M_impl._M_header));
_M_destroy_node(__y);
--_M_impl._M_node_count;
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
void
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
_M_erase_aux(const_iterator __first, const_iterator __last)
{
if (__first == begin() && __last == end())
clear();
else
while (__first != __last)
erase(__first++);
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::size_type
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
erase(const _Key& __x)
{
pair<iterator, iterator> __p = equal_range(__x);
const size_type __old_size = size();
erase(__p.first, __p.second);
return __old_size - size();
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
void
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
erase(const _Key* __first, const _Key* __last)
{
while (__first != __last)
erase(*__first++);
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::iterator
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
find(const _Key& __k)
{
iterator __j = _M_lower_bound(_M_begin(), _M_end(), __k);
return (__j == end()
|| _M_impl._M_key_compare(__k,
_S_key(__j._M_node))) ? end() : __j;
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
typename _Rb_tree<_Key, _Val, _KeyOfValue,
_Compare, _Alloc>::const_iterator
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
find(const _Key& __k) const
{
const_iterator __j = _M_lower_bound(_M_begin(), _M_end(), __k);
return (__j == end()
|| _M_impl._M_key_compare(__k,
_S_key(__j._M_node))) ? end() : __j;
}
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::size_type
_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
count(const _Key& __k) const
{
pair<const_iterator, const_iterator> __p = equal_range(__k);
const size_type __n = std::distance(__p.first, __p.second);
return __n;
}
__attribute__ ((__pure__)) unsigned int
_Rb_tree_black_count(const _Rb_tree_node_base* __node,
const _Rb_tree_node_base* __root) throw ();
template<typename _Key, typename _Val, typename _KeyOfValue,
typename _Compare, typename _Alloc>
bool
_Rb_tree<_Key,_Val,_KeyOfValue,_Compare,_Alloc>::__rb_verify() const
{
if (_M_impl._M_node_count == 0 || begin() == end())
return _M_impl._M_node_count == 0 && begin() == end()
&& this->_M_impl._M_header._M_left == _M_end()
&& this->_M_impl._M_header._M_right == _M_end();
unsigned int __len = _Rb_tree_black_count(_M_leftmost(), _M_root());
for (const_iterator __it = begin(); __it != end(); ++__it)
{
_Const_Link_type __x = static_cast<_Const_Link_type>(__it._M_node);
_Const_Link_type __L = _S_left(__x);
_Const_Link_type __R = _S_right(__x);
if (__x->_M_color == _S_red)
if ((__L && __L->_M_color == _S_red)
|| (__R && __R->_M_color == _S_red))
return false;
if (__L && _M_impl._M_key_compare(_S_key(__x), _S_key(__L)))
return false;
if (__R && _M_impl._M_key_compare(_S_key(__R), _S_key(__x)))
return false;
if (!__L && !__R && _Rb_tree_black_count(__x, _M_root()) != __len)
return false;
}
if (_M_leftmost() != _Rb_tree_node_base::_S_minimum(_M_root()))
return false;
if (_M_rightmost() != _Rb_tree_node_base::_S_maximum(_M_root()))
return false;
return true;
}
}
namespace std __attribute__ ((__visibility__ ("default")))
{
template<typename _Key, typename _Compare = std::less<_Key>,
typename _Alloc = std::allocator<_Key> >
class set
{
typedef typename _Alloc::value_type _Alloc_value_type;
public:
typedef _Key key_type;
typedef _Key value_type;
typedef _Compare key_compare;
typedef _Compare value_compare;
typedef _Alloc allocator_type;
private:
typedef typename _Alloc::template rebind<_Key>::other _Key_alloc_type;
typedef _Rb_tree<key_type, value_type, _Identity<value_type>,
key_compare, _Key_alloc_type> _Rep_type;
_Rep_type _M_t;
public:
typedef typename _Key_alloc_type::pointer pointer;
typedef typename _Key_alloc_type::const_pointer const_pointer;
typedef typename _Key_alloc_type::reference reference;
typedef typename _Key_alloc_type::const_reference const_reference;
typedef typename _Rep_type::const_iterator iterator;
typedef typename _Rep_type::const_iterator const_iterator;
typedef typename _Rep_type::const_reverse_iterator reverse_iterator;
typedef typename _Rep_type::const_reverse_iterator const_reverse_iterator;
typedef typename _Rep_type::size_type size_type;
typedef typename _Rep_type::difference_type difference_type;
set()
: _M_t() { }
explicit
set(const _Compare& __comp,
const allocator_type& __a = allocator_type())
: _M_t(__comp, _Key_alloc_type(__a)) { }
template<typename _InputIterator>
set(_InputIterator __first, _InputIterator __last)
: _M_t()
{ _M_t._M_insert_unique(__first, __last); }
template<typename _InputIterator>
set(_InputIterator __first, _InputIterator __last,
const _Compare& __comp,
const allocator_type& __a = allocator_type())
: _M_t(__comp, _Key_alloc_type(__a))
{ _M_t._M_insert_unique(__first, __last); }
set(const set& __x)
: _M_t(__x._M_t) { }
set&
operator=(const set& __x)
{
_M_t = __x._M_t;
return *this;
}
key_compare
key_comp() const
{ return _M_t.key_comp(); }
value_compare
value_comp() const
{ return _M_t.key_comp(); }
allocator_type
get_allocator() const
{ return allocator_type(_M_t.get_allocator()); }
iterator
begin() const
{ return _M_t.begin(); }
iterator
end() const
{ return _M_t.end(); }
reverse_iterator
rbegin() const
{ return _M_t.rbegin(); }
reverse_iterator
rend() const
{ return _M_t.rend(); }
bool
empty() const
{ return _M_t.empty(); }
size_type
size() const
{ return _M_t.size(); }
size_type
max_size() const
{ return _M_t.max_size(); }
void
swap(set& __x)
{ _M_t.swap(__x._M_t); }
std::pair<iterator, bool>
insert(const value_type& __x)
{
std::pair<typename _Rep_type::iterator, bool> __p =
_M_t._M_insert_unique(__x);
return std::pair<iterator, bool>(__p.first, __p.second);
}
iterator
insert(const_iterator __position, const value_type& __x)
{ return _M_t._M_insert_unique_(__position, __x); }
template<typename _InputIterator>
void
insert(_InputIterator __first, _InputIterator __last)
{ _M_t._M_insert_unique(__first, __last); }
void
erase(iterator __position)
{ _M_t.erase(__position); }
size_type
erase(const key_type& __x)
{ return _M_t.erase(__x); }
void
erase(iterator __first, iterator __last)
{ _M_t.erase(__first, __last); }
void
clear()
{ _M_t.clear(); }
size_type
count(const key_type& __x) const
{ return _M_t.find(__x) == _M_t.end() ? 0 : 1; }
iterator
find(const key_type& __x)
{ return _M_t.find(__x); }
const_iterator
find(const key_type& __x) const
{ return _M_t.find(__x); }
iterator
lower_bound(const key_type& __x)
{ return _M_t.lower_bound(__x); }
const_iterator
lower_bound(const key_type& __x) const
{ return _M_t.lower_bound(__x); }
iterator
upper_bound(const key_type& __x)
{ return _M_t.upper_bound(__x); }
const_iterator
upper_bound(const key_type& __x) const
{ return _M_t.upper_bound(__x); }
std::pair<iterator, iterator>
equal_range(const key_type& __x)
{ return _M_t.equal_range(__x); }
std::pair<const_iterator, const_iterator>
equal_range(const key_type& __x) const
{ return _M_t.equal_range(__x); }
template<typename _K1, typename _C1, typename _A1>
friend bool
operator==(const set<_K1, _C1, _A1>&, const set<_K1, _C1, _A1>&);
template<typename _K1, typename _C1, typename _A1>
friend bool
operator<(const set<_K1, _C1, _A1>&, const set<_K1, _C1, _A1>&);
};
template<typename _Key, typename _Compare, typename _Alloc>
inline bool
operator==(const set<_Key, _Compare, _Alloc>& __x,
const set<_Key, _Compare, _Alloc>& __y)
{ return __x._M_t == __y._M_t; }
template<typename _Key, typename _Compare, typename _Alloc>
inline bool
operator<(const set<_Key, _Compare, _Alloc>& __x,
const set<_Key, _Compare, _Alloc>& __y)
{ return __x._M_t < __y._M_t; }
template<typename _Key, typename _Compare, typename _Alloc>
inline bool
operator!=(const set<_Key, _Compare, _Alloc>& __x,
const set<_Key, _Compare, _Alloc>& __y)
{ return !(__x == __y); }
template<typename _Key, typename _Compare, typename _Alloc>
inline bool
operator>(const set<_Key, _Compare, _Alloc>& __x,
const set<_Key, _Compare, _Alloc>& __y)
{ return __y < __x; }
template<typename _Key, typename _Compare, typename _Alloc>
inline bool
operator<=(const set<_Key, _Compare, _Alloc>& __x,
const set<_Key, _Compare, _Alloc>& __y)
{ return !(__y < __x); }
template<typename _Key, typename _Compare, typename _Alloc>
inline bool
operator>=(const set<_Key, _Compare, _Alloc>& __x,
const set<_Key, _Compare, _Alloc>& __y)
{ return !(__x < __y); }
template<typename _Key, typename _Compare, typename _Alloc>
inline void
swap(set<_Key, _Compare, _Alloc>& __x, set<_Key, _Compare, _Alloc>& __y)
{ __x.swap(__y); }
}
namespace std __attribute__ ((__visibility__ ("default")))
{
template <typename _Key, typename _Compare = std::less<_Key>,
typename _Alloc = std::allocator<_Key> >
class multiset
{
typedef typename _Alloc::value_type _Alloc_value_type;
public:
typedef _Key key_type;
typedef _Key value_type;
typedef _Compare key_compare;
typedef _Compare value_compare;
typedef _Alloc allocator_type;
private:
typedef typename _Alloc::template rebind<_Key>::other _Key_alloc_type;
typedef _Rb_tree<key_type, value_type, _Identity<value_type>,
key_compare, _Key_alloc_type> _Rep_type;
_Rep_type _M_t;
public:
typedef typename _Key_alloc_type::pointer pointer;
typedef typename _Key_alloc_type::const_pointer const_pointer;
typedef typename _Key_alloc_type::reference reference;
typedef typename _Key_alloc_type::const_reference const_reference;
typedef typename _Rep_type::const_iterator iterator;
typedef typename _Rep_type::const_iterator const_iterator;
typedef typename _Rep_type::const_reverse_iterator reverse_iterator;
typedef typename _Rep_type::const_reverse_iterator const_reverse_iterator;
typedef typename _Rep_type::size_type size_type;
typedef typename _Rep_type::difference_type difference_type;
multiset()
: _M_t() { }
explicit
multiset(const _Compare& __comp,
const allocator_type& __a = allocator_type())
: _M_t(__comp, _Key_alloc_type(__a)) { }
template<typename _InputIterator>
multiset(_InputIterator __first, _InputIterator __last)
: _M_t()
{ _M_t._M_insert_equal(__first, __last); }
template<typename _InputIterator>
multiset(_InputIterator __first, _InputIterator __last,
const _Compare& __comp,
const allocator_type& __a = allocator_type())
: _M_t(__comp, _Key_alloc_type(__a))
{ _M_t._M_insert_equal(__first, __last); }
multiset(const multiset& __x)
: _M_t(__x._M_t) { }
multiset&
operator=(const multiset& __x)
{
_M_t = __x._M_t;
return *this;
}
key_compare
key_comp() const
{ return _M_t.key_comp(); }
value_compare
value_comp() const
{ return _M_t.key_comp(); }
allocator_type
get_allocator() const
{ return allocator_type(_M_t.get_allocator()); }
iterator
begin() const
{ return _M_t.begin(); }
iterator
end() const
{ return _M_t.end(); }
reverse_iterator
rbegin() const
{ return _M_t.rbegin(); }
reverse_iterator
rend() const
{ return _M_t.rend(); }
bool
empty() const
{ return _M_t.empty(); }
size_type
size() const
{ return _M_t.size(); }
size_type
max_size() const
{ return _M_t.max_size(); }
void
swap(multiset& __x)
{ _M_t.swap(__x._M_t); }
iterator
insert(const value_type& __x)
{ return _M_t._M_insert_equal(__x); }
iterator
insert(const_iterator __position, const value_type& __x)
{ return _M_t._M_insert_equal_(__position, __x); }
template<typename _InputIterator>
void
insert(_InputIterator __first, _InputIterator __last)
{ _M_t._M_insert_equal(__first, __last); }
void
erase(iterator __position)
{ _M_t.erase(__position); }
size_type
erase(const key_type& __x)
{ return _M_t.erase(__x); }
void
erase(iterator __first, iterator __last)
{ _M_t.erase(__first, __last); }
void
clear()
{ _M_t.clear(); }
size_type
count(const key_type& __x) const
{ return _M_t.count(__x); }
iterator
find(const key_type& __x)
{ return _M_t.find(__x); }
const_iterator
find(const key_type& __x) const
{ return _M_t.find(__x); }
iterator
lower_bound(const key_type& __x)
{ return _M_t.lower_bound(__x); }
const_iterator
lower_bound(const key_type& __x) const
{ return _M_t.lower_bound(__x); }
iterator
upper_bound(const key_type& __x)
{ return _M_t.upper_bound(__x); }
const_iterator
upper_bound(const key_type& __x) const
{ return _M_t.upper_bound(__x); }
std::pair<iterator, iterator>
equal_range(const key_type& __x)
{ return _M_t.equal_range(__x); }
std::pair<const_iterator, const_iterator>
equal_range(const key_type& __x) const
{ return _M_t.equal_range(__x); }
template<typename _K1, typename _C1, typename _A1>
friend bool
operator==(const multiset<_K1, _C1, _A1>&,
const multiset<_K1, _C1, _A1>&);
template<typename _K1, typename _C1, typename _A1>
friend bool
operator< (const multiset<_K1, _C1, _A1>&,
const multiset<_K1, _C1, _A1>&);
};
template<typename _Key, typename _Compare, typename _Alloc>
inline bool
operator==(const multiset<_Key, _Compare, _Alloc>& __x,
const multiset<_Key, _Compare, _Alloc>& __y)
{ return __x._M_t == __y._M_t; }
template<typename _Key, typename _Compare, typename _Alloc>
inline bool
operator<(const multiset<_Key, _Compare, _Alloc>& __x,
const multiset<_Key, _Compare, _Alloc>& __y)
{ return __x._M_t < __y._M_t; }
template<typename _Key, typename _Compare, typename _Alloc>
inline bool
operator!=(const multiset<_Key, _Compare, _Alloc>& __x,
const multiset<_Key, _Compare, _Alloc>& __y)
{ return !(__x == __y); }
template<typename _Key, typename _Compare, typename _Alloc>
inline bool
operator>(const multiset<_Key,_Compare,_Alloc>& __x,
const multiset<_Key,_Compare,_Alloc>& __y)
{ return __y < __x; }
template<typename _Key, typename _Compare, typename _Alloc>
inline bool
operator<=(const multiset<_Key, _Compare, _Alloc>& __x,
const multiset<_Key, _Compare, _Alloc>& __y)
{ return !(__y < __x); }
template<typename _Key, typename _Compare, typename _Alloc>
inline bool
operator>=(const multiset<_Key, _Compare, _Alloc>& __x,
const multiset<_Key, _Compare, _Alloc>& __y)
{ return !(__x < __y); }
template<typename _Key, typename _Compare, typename _Alloc>
inline void
swap(multiset<_Key, _Compare, _Alloc>& __x,
multiset<_Key, _Compare, _Alloc>& __y)
{ __x.swap(__y); }
}
#endif
