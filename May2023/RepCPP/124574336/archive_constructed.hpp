

#ifndef BOOST_MULTI_INDEX_DETAIL_ARCHIVE_CONSTRUCTED_HPP
#define BOOST_MULTI_INDEX_DETAIL_ARCHIVE_CONSTRUCTED_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/core/no_exceptions_support.hpp>
#include <boost/noncopyable.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/type_traits/aligned_storage.hpp>
#include <boost/type_traits/alignment_of.hpp> 

namespace boost{

namespace multi_index{

namespace detail{



template<typename T>
struct archive_constructed:private noncopyable
{
template<class Archive>
archive_constructed(Archive& ar,const unsigned int version)
{
serialization::load_construct_data_adl(ar,&get(),version);
BOOST_TRY{
ar>>get();
}
BOOST_CATCH(...){
(&get())->~T();
BOOST_RETHROW;
}
BOOST_CATCH_END
}

template<class Archive>
archive_constructed(const char* name,Archive& ar,const unsigned int version)
{
serialization::load_construct_data_adl(ar,&get(),version);
BOOST_TRY{
ar>>serialization::make_nvp(name,get());
}
BOOST_CATCH(...){
(&get())->~T();
BOOST_RETHROW;
}
BOOST_CATCH_END
}

~archive_constructed()
{
(&get())->~T();
}

#include <boost/multi_index/detail/ignore_wstrict_aliasing.hpp>

T& get(){return *reinterpret_cast<T*>(&space);}

#include <boost/multi_index/detail/restore_wstrict_aliasing.hpp>

private:
typename aligned_storage<sizeof(T),alignment_of<T>::value>::type space;
};

} 

} 

} 

#endif
