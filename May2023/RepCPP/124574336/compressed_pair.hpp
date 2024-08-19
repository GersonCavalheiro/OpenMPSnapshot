

#ifndef BOOST_DETAIL_COMPRESSED_PAIR_HPP
#define BOOST_DETAIL_COMPRESSED_PAIR_HPP

#include <algorithm>

#include <boost/type_traits/remove_cv.hpp>
#include <boost/type_traits/is_empty.hpp>
#include <boost/type_traits/is_final.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/call_traits.hpp>

#ifdef BOOST_MSVC
# pragma warning(push)
# pragma warning(disable:4512)
#endif 
namespace boost
{

template <class T1, class T2>
class compressed_pair;



namespace details
{
template<class T, bool E = boost::is_final<T>::value>
struct compressed_pair_empty
: ::boost::false_type { };

template<class T>
struct compressed_pair_empty<T, false>
: ::boost::is_empty<T> { };

template <class T1, class T2, bool IsSame, bool FirstEmpty, bool SecondEmpty>
struct compressed_pair_switch;

template <class T1, class T2>
struct compressed_pair_switch<T1, T2, false, false, false>
{static const int value = 0;};

template <class T1, class T2>
struct compressed_pair_switch<T1, T2, false, true, true>
{static const int value = 3;};

template <class T1, class T2>
struct compressed_pair_switch<T1, T2, false, true, false>
{static const int value = 1;};

template <class T1, class T2>
struct compressed_pair_switch<T1, T2, false, false, true>
{static const int value = 2;};

template <class T1, class T2>
struct compressed_pair_switch<T1, T2, true, true, true>
{static const int value = 4;};

template <class T1, class T2>
struct compressed_pair_switch<T1, T2, true, false, false>
{static const int value = 5;};

template <class T1, class T2, int Version> class compressed_pair_imp;

#ifdef __GNUC__
using std::swap;
#endif
template <typename T>
inline void cp_swap(T& t1, T& t2)
{
#ifndef __GNUC__
using std::swap;
#endif
swap(t1, t2);
}


template <class T1, class T2>
class compressed_pair_imp<T1, T2, 0>
{
public:
typedef T1                                                 first_type;
typedef T2                                                 second_type;
typedef typename call_traits<first_type>::param_type       first_param_type;
typedef typename call_traits<second_type>::param_type      second_param_type;
typedef typename call_traits<first_type>::reference        first_reference;
typedef typename call_traits<second_type>::reference       second_reference;
typedef typename call_traits<first_type>::const_reference  first_const_reference;
typedef typename call_traits<second_type>::const_reference second_const_reference;

compressed_pair_imp() {} 

compressed_pair_imp(first_param_type x, second_param_type y)
: first_(x), second_(y) {}

compressed_pair_imp(first_param_type x)
: first_(x) {}

compressed_pair_imp(second_param_type y)
: second_(y) {}

first_reference       first()       {return first_;}
first_const_reference first() const {return first_;}

second_reference       second()       {return second_;}
second_const_reference second() const {return second_;}

void swap(::boost::compressed_pair<T1, T2>& y)
{
cp_swap(first_, y.first());
cp_swap(second_, y.second());
}
private:
first_type first_;
second_type second_;
};


template <class T1, class T2>
class compressed_pair_imp<T1, T2, 1>
: protected ::boost::remove_cv<T1>::type
{
public:
typedef T1                                                 first_type;
typedef T2                                                 second_type;
typedef typename call_traits<first_type>::param_type       first_param_type;
typedef typename call_traits<second_type>::param_type      second_param_type;
typedef typename call_traits<first_type>::reference        first_reference;
typedef typename call_traits<second_type>::reference       second_reference;
typedef typename call_traits<first_type>::const_reference  first_const_reference;
typedef typename call_traits<second_type>::const_reference second_const_reference;

compressed_pair_imp() {}

compressed_pair_imp(first_param_type x, second_param_type y)
: first_type(x), second_(y) {}

compressed_pair_imp(first_param_type x)
: first_type(x) {}

compressed_pair_imp(second_param_type y)
: second_(y) {}

first_reference       first()       {return *this;}
first_const_reference first() const {return *this;}

second_reference       second()       {return second_;}
second_const_reference second() const {return second_;}

void swap(::boost::compressed_pair<T1,T2>& y)
{
cp_swap(second_, y.second());
}
private:
second_type second_;
};


template <class T1, class T2>
class compressed_pair_imp<T1, T2, 2>
: protected ::boost::remove_cv<T2>::type
{
public:
typedef T1                                                 first_type;
typedef T2                                                 second_type;
typedef typename call_traits<first_type>::param_type       first_param_type;
typedef typename call_traits<second_type>::param_type      second_param_type;
typedef typename call_traits<first_type>::reference        first_reference;
typedef typename call_traits<second_type>::reference       second_reference;
typedef typename call_traits<first_type>::const_reference  first_const_reference;
typedef typename call_traits<second_type>::const_reference second_const_reference;

compressed_pair_imp() {}

compressed_pair_imp(first_param_type x, second_param_type y)
: second_type(y), first_(x) {}

compressed_pair_imp(first_param_type x)
: first_(x) {}

compressed_pair_imp(second_param_type y)
: second_type(y) {}

first_reference       first()       {return first_;}
first_const_reference first() const {return first_;}

second_reference       second()       {return *this;}
second_const_reference second() const {return *this;}

void swap(::boost::compressed_pair<T1,T2>& y)
{
cp_swap(first_, y.first());
}

private:
first_type first_;
};


template <class T1, class T2>
class compressed_pair_imp<T1, T2, 3>
: protected ::boost::remove_cv<T1>::type,
protected ::boost::remove_cv<T2>::type
{
public:
typedef T1                                                 first_type;
typedef T2                                                 second_type;
typedef typename call_traits<first_type>::param_type       first_param_type;
typedef typename call_traits<second_type>::param_type      second_param_type;
typedef typename call_traits<first_type>::reference        first_reference;
typedef typename call_traits<second_type>::reference       second_reference;
typedef typename call_traits<first_type>::const_reference  first_const_reference;
typedef typename call_traits<second_type>::const_reference second_const_reference;

compressed_pair_imp() {}

compressed_pair_imp(first_param_type x, second_param_type y)
: first_type(x), second_type(y) {}

compressed_pair_imp(first_param_type x)
: first_type(x) {}

compressed_pair_imp(second_param_type y)
: second_type(y) {}

first_reference       first()       {return *this;}
first_const_reference first() const {return *this;}

second_reference       second()       {return *this;}
second_const_reference second() const {return *this;}
void swap(::boost::compressed_pair<T1,T2>&) {}
};

template <class T1, class T2>
class compressed_pair_imp<T1, T2, 4>
: protected ::boost::remove_cv<T1>::type
{
public:
typedef T1                                                 first_type;
typedef T2                                                 second_type;
typedef typename call_traits<first_type>::param_type       first_param_type;
typedef typename call_traits<second_type>::param_type      second_param_type;
typedef typename call_traits<first_type>::reference        first_reference;
typedef typename call_traits<second_type>::reference       second_reference;
typedef typename call_traits<first_type>::const_reference  first_const_reference;
typedef typename call_traits<second_type>::const_reference second_const_reference;

compressed_pair_imp() {}

compressed_pair_imp(first_param_type x, second_param_type y)
: first_type(x), m_second(y) {}

compressed_pair_imp(first_param_type x)
: first_type(x), m_second(x) {}

first_reference       first()       {return *this;}
first_const_reference first() const {return *this;}

second_reference       second()       {return m_second;}
second_const_reference second() const {return m_second;}

void swap(::boost::compressed_pair<T1,T2>&) {}
private:
T2 m_second;
};


template <class T1, class T2>
class compressed_pair_imp<T1, T2, 5>
{
public:
typedef T1                                                 first_type;
typedef T2                                                 second_type;
typedef typename call_traits<first_type>::param_type       first_param_type;
typedef typename call_traits<second_type>::param_type      second_param_type;
typedef typename call_traits<first_type>::reference        first_reference;
typedef typename call_traits<second_type>::reference       second_reference;
typedef typename call_traits<first_type>::const_reference  first_const_reference;
typedef typename call_traits<second_type>::const_reference second_const_reference;

compressed_pair_imp() {}

compressed_pair_imp(first_param_type x, second_param_type y)
: first_(x), second_(y) {}

compressed_pair_imp(first_param_type x)
: first_(x), second_(x) {}

first_reference       first()       {return first_;}
first_const_reference first() const {return first_;}

second_reference       second()       {return second_;}
second_const_reference second() const {return second_;}

void swap(::boost::compressed_pair<T1, T2>& y)
{
cp_swap(first_, y.first());
cp_swap(second_, y.second());
}
private:
first_type first_;
second_type second_;
};

}  

template <class T1, class T2>
class compressed_pair
: private ::boost::details::compressed_pair_imp<T1, T2,
::boost::details::compressed_pair_switch<
T1,
T2,
::boost::is_same<typename remove_cv<T1>::type, typename remove_cv<T2>::type>::value,
::boost::details::compressed_pair_empty<T1>::value,
::boost::details::compressed_pair_empty<T2>::value>::value>
{
private:
typedef details::compressed_pair_imp<T1, T2,
::boost::details::compressed_pair_switch<
T1,
T2,
::boost::is_same<typename remove_cv<T1>::type, typename remove_cv<T2>::type>::value,
::boost::details::compressed_pair_empty<T1>::value,
::boost::details::compressed_pair_empty<T2>::value>::value> base;
public:
typedef T1                                                 first_type;
typedef T2                                                 second_type;
typedef typename call_traits<first_type>::param_type       first_param_type;
typedef typename call_traits<second_type>::param_type      second_param_type;
typedef typename call_traits<first_type>::reference        first_reference;
typedef typename call_traits<second_type>::reference       second_reference;
typedef typename call_traits<first_type>::const_reference  first_const_reference;
typedef typename call_traits<second_type>::const_reference second_const_reference;

compressed_pair() : base() {}
compressed_pair(first_param_type x, second_param_type y) : base(x, y) {}
explicit compressed_pair(first_param_type x) : base(x) {}
explicit compressed_pair(second_param_type y) : base(y) {}

first_reference       first()       {return base::first();}
first_const_reference first() const {return base::first();}

second_reference       second()       {return base::second();}
second_const_reference second() const {return base::second();}

void swap(compressed_pair& y) { base::swap(y); }
};

template <class T>
class compressed_pair<T, T>
: private details::compressed_pair_imp<T, T,
::boost::details::compressed_pair_switch<
T,
T,
::boost::is_same<typename remove_cv<T>::type, typename remove_cv<T>::type>::value,
::boost::details::compressed_pair_empty<T>::value,
::boost::details::compressed_pair_empty<T>::value>::value>
{
private:
typedef details::compressed_pair_imp<T, T,
::boost::details::compressed_pair_switch<
T,
T,
::boost::is_same<typename remove_cv<T>::type, typename remove_cv<T>::type>::value,
::boost::details::compressed_pair_empty<T>::value,
::boost::details::compressed_pair_empty<T>::value>::value> base;
public:
typedef T                                                  first_type;
typedef T                                                  second_type;
typedef typename call_traits<first_type>::param_type       first_param_type;
typedef typename call_traits<second_type>::param_type      second_param_type;
typedef typename call_traits<first_type>::reference        first_reference;
typedef typename call_traits<second_type>::reference       second_reference;
typedef typename call_traits<first_type>::const_reference  first_const_reference;
typedef typename call_traits<second_type>::const_reference second_const_reference;

compressed_pair() : base() {}
compressed_pair(first_param_type x, second_param_type y) : base(x, y) {}
#if !(defined(__SUNPRO_CC) && (__SUNPRO_CC <= 0x530))
explicit 
#endif
compressed_pair(first_param_type x) : base(x) {}

first_reference       first()       {return base::first();}
first_const_reference first() const {return base::first();}

second_reference       second()       {return base::second();}
second_const_reference second() const {return base::second();}

void swap(::boost::compressed_pair<T,T>& y) { base::swap(y); }
};

template <class T1, class T2>
inline
void
swap(compressed_pair<T1, T2>& x, compressed_pair<T1, T2>& y)
{
x.swap(y);
}

} 

#ifdef BOOST_MSVC
# pragma warning(pop)
#endif 

#endif 

