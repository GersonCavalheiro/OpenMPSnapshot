


#ifndef BOOST_MPI_DETAIL_TYPE_MPI_DATATYPE_CACHE_HPP
#define BOOST_MPI_DETAIL_TYPE_MPI_DATATYPE_CACHE_HPP

#include <boost/mpi/datatype_fwd.hpp>
#include <boost/mpi/detail/mpi_datatype_oarchive.hpp>
#include <boost/mpi/exception.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/noncopyable.hpp>
#include <typeinfo>

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4800)
#endif

namespace boost { namespace mpi { namespace detail {


struct type_info_compare
{
bool operator()(std::type_info const* lhs, std::type_info const* rhs) const
{
return lhs->before(*rhs);
}
};


class BOOST_MPI_DECL mpi_datatype_map
: public boost::noncopyable
{
struct implementation;

implementation *impl;

public:
mpi_datatype_map();
~mpi_datatype_map();

template <class T>
MPI_Datatype datatype(const T& x = T(), typename boost::enable_if<is_mpi_builtin_datatype<T> >::type* =0)
{
return get_mpi_datatype<T>(x);
}

template <class T>
MPI_Datatype datatype(const T& x =T(), typename boost::disable_if<is_mpi_builtin_datatype<T> >::type* =0 )
{
BOOST_MPL_ASSERT((is_mpi_datatype<T>));

std::type_info const* t = &typeid(T);
MPI_Datatype datatype = get(t);
if (datatype == MPI_DATATYPE_NULL) {
mpi_datatype_oarchive ar(x);
datatype = ar.get_mpi_datatype();
set(t, datatype);
}

return datatype;
}

void clear(); 

private:
MPI_Datatype get(const std::type_info* t);
void set(const std::type_info* t, MPI_Datatype datatype);
};

BOOST_MPI_DECL mpi_datatype_map& mpi_datatype_cache();

} } } 

#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif

#endif 
