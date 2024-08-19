#ifndef BOOST_SMART_PTR_DETAIL_LOCAL_SP_DELETER_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_LOCAL_SP_DELETER_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/smart_ptr/detail/local_counted_base.hpp>
#include <boost/config.hpp>

namespace boost
{

namespace detail
{

template<class D> class local_sp_deleter: public local_counted_impl_em
{
private:

D d_;

public:

local_sp_deleter(): d_()
{
}

explicit local_sp_deleter( D const& d ) BOOST_SP_NOEXCEPT: d_( d )
{
}

#if !defined( BOOST_NO_CXX11_RVALUE_REFERENCES )

explicit local_sp_deleter( D&& d ) BOOST_SP_NOEXCEPT: d_( std::move(d) )
{
}

#endif

D& deleter() BOOST_SP_NOEXCEPT
{
return d_;
}

template<class Y> void operator()( Y* p ) BOOST_SP_NOEXCEPT
{
d_( p );
}

#if !defined( BOOST_NO_CXX11_NULLPTR )

void operator()( boost::detail::sp_nullptr_t p ) BOOST_SP_NOEXCEPT
{
d_( p );
}

#endif
};

template<> class local_sp_deleter<void>
{
};

template<class D> D * get_local_deleter( local_sp_deleter<D> * p ) BOOST_SP_NOEXCEPT
{
return &p->deleter();
}

inline void * get_local_deleter( local_sp_deleter<void> *  ) BOOST_SP_NOEXCEPT
{
return 0;
}

} 

} 

#endif  
