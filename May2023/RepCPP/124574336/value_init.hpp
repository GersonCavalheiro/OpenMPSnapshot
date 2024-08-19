#ifndef BOOST_UTILITY_VALUE_INIT_21AGO2002_HPP
#define BOOST_UTILITY_VALUE_INIT_21AGO2002_HPP


#include <boost/config.hpp> 
#include <boost/swap.hpp>
#include <cstring>
#include <cstddef>

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4351)
#pragma warning(disable: 4512)
#endif

#ifdef BOOST_NO_COMPLETE_VALUE_INITIALIZATION
#define BOOST_DETAIL_VALUE_INIT_WORKAROUND_SUGGESTED
#endif

#ifndef BOOST_DETAIL_VALUE_INIT_WORKAROUND
#ifdef BOOST_DETAIL_VALUE_INIT_WORKAROUND_SUGGESTED
#define BOOST_DETAIL_VALUE_INIT_WORKAROUND 1
#else
#define BOOST_DETAIL_VALUE_INIT_WORKAROUND 0
#endif
#endif

namespace boost {

namespace detail {

struct zero_init
{
zero_init()
{
}

zero_init( void * p, std::size_t n )
{
std::memset( p, 0, n );
}
};

} 

template<class T>
class initialized
#if BOOST_DETAIL_VALUE_INIT_WORKAROUND
: detail::zero_init
#endif
{
private:

T data_;

public :

BOOST_GPU_ENABLED
initialized():
#if BOOST_DETAIL_VALUE_INIT_WORKAROUND
zero_init( &const_cast< char& >( reinterpret_cast<char const volatile&>( data_ ) ), sizeof( data_ ) ),
#endif
data_()
{
}

BOOST_GPU_ENABLED
explicit initialized(T const & arg): data_( arg )
{
}

BOOST_GPU_ENABLED
T const & data() const
{
return data_;
}

BOOST_GPU_ENABLED
T& data()
{
return data_;
}

BOOST_GPU_ENABLED
void swap(initialized & arg)
{
::boost::swap( this->data(), arg.data() );
}

BOOST_GPU_ENABLED
operator T const &() const
{
return data_;
}

BOOST_GPU_ENABLED
operator T&()
{
return data_;
}

} ;

template<class T>
BOOST_GPU_ENABLED
T const& get ( initialized<T> const& x )
{
return x.data() ;
}

template<class T>
BOOST_GPU_ENABLED
T& get ( initialized<T>& x )
{
return x.data() ;
}

template<class T>
BOOST_GPU_ENABLED
void swap ( initialized<T> & lhs, initialized<T> & rhs )
{
lhs.swap(rhs) ;
}

template<class T>
class value_initialized
{
private :

initialized<T> m_data;

public :

BOOST_GPU_ENABLED
value_initialized()
:
m_data()
{ }

BOOST_GPU_ENABLED
T const & data() const
{
return m_data.data();
}

BOOST_GPU_ENABLED
T& data()
{
return m_data.data();
}

BOOST_GPU_ENABLED
void swap(value_initialized & arg)
{
m_data.swap(arg.m_data);
}

BOOST_GPU_ENABLED
operator T const &() const
{
return m_data;
}

BOOST_GPU_ENABLED
operator T&()
{
return m_data;
}
} ;


template<class T>
BOOST_GPU_ENABLED
T const& get ( value_initialized<T> const& x )
{
return x.data() ;
}

template<class T>
BOOST_GPU_ENABLED
T& get ( value_initialized<T>& x )
{
return x.data() ;
}

template<class T>
BOOST_GPU_ENABLED
void swap ( value_initialized<T> & lhs, value_initialized<T> & rhs )
{
lhs.swap(rhs) ;
}


class initialized_value_t
{
public :

template <class T> BOOST_GPU_ENABLED operator T() const
{
return initialized<T>().data();
}
};

initialized_value_t const initialized_value = {} ;


} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif
