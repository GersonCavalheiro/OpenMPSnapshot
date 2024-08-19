


#ifndef BOOST_MPI_COMMUNICATOR_HPP
#define BOOST_MPI_COMMUNICATOR_HPP

#include <boost/assert.hpp>
#include <boost/mpi/config.hpp>
#include <boost/mpi/exception.hpp>
#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/mpi/datatype.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <boost/static_assert.hpp>
#include <utility>
#include <iterator>
#include <stdexcept> 
#include <vector>

#include <boost/mpi/packed_oarchive.hpp>
#include <boost/mpi/packed_iarchive.hpp>

#include <boost/mpi/skeleton_and_content_fwd.hpp>

#include <boost/mpi/detail/point_to_point.hpp>
#include <boost/mpi/status.hpp>
#include <boost/mpi/request.hpp>

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4800) 
#endif

namespace boost { namespace mpi {


const int any_source = MPI_ANY_SOURCE;


const int any_tag = MPI_ANY_TAG;


enum comm_create_kind { comm_duplicate, comm_take_ownership, comm_attach };


class group;


class intercommunicator;


class graph_communicator;


class cartesian_communicator;


class BOOST_MPI_DECL communicator
{
public:

communicator();


communicator(const MPI_Comm& comm, comm_create_kind kind);


communicator(const communicator& comm, const boost::mpi::group& subgroup);


int rank() const;


int size() const;


boost::mpi::group group() const;



template<typename T>
void send(int dest, int tag, const T& value) const;

template<typename T, typename A>
void send(int dest, int tag, const std::vector<T,A>& value) const;


template<typename T>
void send(int dest, int tag, const skeleton_proxy<T>& proxy) const;


template<typename T>
void send(int dest, int tag, const T* values, int n) const;


void send(int dest, int tag) const;


template<typename T>
status recv(int source, int tag, T& value) const;

template<typename T, typename A>
status recv(int source, int tag, std::vector<T,A>& value) const;


template<typename T>
status recv(int source, int tag, const skeleton_proxy<T>& proxy) const;


template<typename T>
status recv(int source, int tag, skeleton_proxy<T>& proxy) const;


template<typename T>
status recv(int source, int tag, T* values, int n) const;


status recv(int source, int tag) const;


template<typename T>
status sendrecv(int dest, int stag, const T& sval, int src, int rtag, T& rval) const;


template<typename T>
request isend(int dest, int tag, const T& value) const;


template<typename T>
request isend(int dest, int tag, const skeleton_proxy<T>& proxy) const;


template<typename T>
request isend(int dest, int tag, const T* values, int n) const;

template<typename T, class A>
request isend(int dest, int tag, const std::vector<T,A>& values) const;


request isend(int dest, int tag) const;


template<typename T>
request irecv(int source, int tag, T& value) const;


template<typename T>
request irecv(int source, int tag, T* values, int n) const;

template<typename T, typename A>
request irecv(int source, int tag, std::vector<T,A>& values) const;


request irecv(int source, int tag) const;


status probe(int source = any_source, int tag = any_tag) const;


optional<status>
iprobe(int source = any_source, int tag = any_tag) const;

#ifdef barrier
void (barrier)() const;
#else

void barrier() const;
#endif


operator bool() const { return (bool)comm_ptr; }


operator MPI_Comm() const;


communicator split(int color, int key) const;
communicator split(int color) const;


optional<intercommunicator> as_intercommunicator() const;


optional<graph_communicator> as_graph_communicator() const;


bool has_graph_topology() const;


optional<cartesian_communicator> as_cartesian_communicator() const;


bool has_cartesian_topology() const;


void abort(int errcode) const;

protected:


template<typename T>
status sendrecv_impl(int dest, int stag, const T& sval, int src, int rtag, T& rval,
mpl::true_) const;


template<typename T>
status sendrecv_impl(int dest, int stag, const T& sval, int src, int rtag, T& rval,
mpl::false_) const;


struct comm_free
{
void operator()(MPI_Comm* comm) const
{
BOOST_ASSERT( comm != 0 );
BOOST_ASSERT(*comm != MPI_COMM_NULL);
int finalized;
BOOST_MPI_CHECK_RESULT(MPI_Finalized, (&finalized));
if (!finalized)
BOOST_MPI_CHECK_RESULT(MPI_Comm_free, (comm));
delete comm;
}
};



template<typename T>
void send_impl(int dest, int tag, const T& value, mpl::true_) const;


template<typename T>
void send_impl(int dest, int tag, const T& value, mpl::false_) const;


template<typename T>
void 
array_send_impl(int dest, int tag, const T* values, int n, mpl::true_) const;


template<typename T>
void 
array_send_impl(int dest, int tag, const T* values, int n, 
mpl::false_) const;


template<typename T>
request isend_impl(int dest, int tag, const T& value, mpl::true_) const;


template<typename T>
request isend_impl(int dest, int tag, const T& value, mpl::false_) const;


template<typename T>
request 
array_isend_impl(int dest, int tag, const T* values, int n, 
mpl::true_) const;


template<typename T>
request 
array_isend_impl(int dest, int tag, const T* values, int n, 
mpl::false_) const;


template<typename T>
status recv_impl(int source, int tag, T& value, mpl::true_) const;


template<typename T>
status recv_impl(int source, int tag, T& value, mpl::false_) const;


template<typename T>
status 
array_recv_impl(int source, int tag, T* values, int n, mpl::true_) const;


template<typename T>
status 
array_recv_impl(int source, int tag, T* values, int n, mpl::false_) const;


template<typename T>
request irecv_impl(int source, int tag, T& value, mpl::true_) const;


template<typename T>
request irecv_impl(int source, int tag, T& value, mpl::false_) const;


template<typename T>
request 
array_irecv_impl(int source, int tag, T* values, int n, mpl::true_) const;


template<typename T>
request 
array_irecv_impl(int source, int tag, T* values, int n, mpl::false_) const;

template<typename T, typename A>
request irecv_vector(int source, int tag, std::vector<T,A>& values, 
mpl::true_) const;
template<typename T, class A>
request isend_vector(int dest, int tag, const std::vector<T,A>& values,
mpl::true_) const;
template<typename T, typename A>
void send_vector(int dest, int tag, const std::vector<T,A>& value, 
mpl::true_) const;
template<typename T, typename A>
status recv_vector(int source, int tag, std::vector<T,A>& value,
mpl::true_) const;

template<typename T, typename A>
request irecv_vector(int source, int tag, std::vector<T,A>& values, 
mpl::false_) const;
template<typename T, class A>
request isend_vector(int dest, int tag, const std::vector<T,A>& values,
mpl::false_) const;
template<typename T, typename A>
void send_vector(int dest, int tag, const std::vector<T,A>& value, 
mpl::false_) const;
template<typename T, typename A>
status recv_vector(int source, int tag, std::vector<T,A>& value,
mpl::false_) const;

protected:
shared_ptr<MPI_Comm> comm_ptr;
};


BOOST_MPI_DECL bool operator==(const communicator& comm1, const communicator& comm2);


inline bool operator!=(const communicator& comm1, const communicator& comm2)
{
return !(comm1 == comm2);
}

}} 



#include <boost/mpi/detail/request_handlers.hpp>

namespace boost { namespace mpi {

template<>
BOOST_MPI_DECL void
communicator::send<packed_oarchive>(int dest, int tag,
const packed_oarchive& ar) const;


template<>
BOOST_MPI_DECL void
communicator::send<packed_skeleton_oarchive>
(int dest, int tag, const packed_skeleton_oarchive& ar) const;


template<>
BOOST_MPI_DECL void
communicator::send<content>(int dest, int tag, const content& c) const;


template<>
BOOST_MPI_DECL status
communicator::recv<packed_iarchive>(int source, int tag,
packed_iarchive& ar) const;


template<>
BOOST_MPI_DECL status
communicator::recv<packed_skeleton_iarchive>
(int source, int tag, packed_skeleton_iarchive& ar) const;


template<>
BOOST_MPI_DECL status
communicator::recv<const content>(int source, int tag,
const content& c) const;


template<>
inline status
communicator::recv<content>(int source, int tag,
content& c) const
{
return recv<const content>(source,tag,c);
}                                  


template<>
BOOST_MPI_DECL request
communicator::isend<packed_oarchive>(int dest, int tag,
const packed_oarchive& ar) const;


template<>
BOOST_MPI_DECL request
communicator::isend<packed_skeleton_oarchive>
(int dest, int tag, const packed_skeleton_oarchive& ar) const;


template<>
BOOST_MPI_DECL request
communicator::isend<content>(int dest, int tag, const content& c) const;


template<>
BOOST_MPI_DECL request
communicator::irecv<packed_skeleton_iarchive>
(int source, int tag, packed_skeleton_iarchive& ar) const;


template<>
BOOST_MPI_DECL request
communicator::irecv<const content>(int source, int tag,
const content& c) const;


template<>
inline request
communicator::irecv<content>(int source, int tag,
content& c) const
{
return irecv<const content>(source, tag, c);
}

template<typename T>
void
communicator::send_impl(int dest, int tag, const T& value, mpl::true_) const
{
BOOST_MPI_CHECK_RESULT(MPI_Send,
(const_cast<T*>(&value), 1, get_mpi_datatype<T>(value),
dest, tag, MPI_Comm(*this)));
}

template<typename T>
void
communicator::send_impl(int dest, int tag, const T& value, mpl::false_) const
{
packed_oarchive oa(*this);
oa << value;
send(dest, tag, oa);
}

template<typename T>
void communicator::send(int dest, int tag, const T& value) const
{
this->send_impl(dest, tag, value, is_mpi_datatype<T>());
}

template<typename T>
void
communicator::array_send_impl(int dest, int tag, const T* values, int n,
mpl::true_) const
{
BOOST_MPI_CHECK_RESULT(MPI_Send,
(const_cast<T*>(values), n, 
get_mpi_datatype<T>(*values),
dest, tag, MPI_Comm(*this)));
}

template<typename T>
void
communicator::array_send_impl(int dest, int tag, const T* values, int n,
mpl::false_) const
{
packed_oarchive oa(*this);
T const* v = values;
while (v < values+n) {
oa << *v++;
}
send(dest, tag, oa);
}

template<typename T, typename A>
void communicator::send_vector(int dest, int tag, 
const std::vector<T,A>& values, mpl::true_ primitive) const
{
#if defined(BOOST_MPI_USE_IMPROBE)
array_send_impl(dest, tag, values.data(), values.size(), primitive);
#else
{
typename std::vector<T,A>::size_type size = values.size();
send(dest, tag, size);
this->array_send_impl(dest, tag, values.data(), size, primitive);
}
#endif
}

template<typename T, typename A>
void communicator::send_vector(int dest, int tag, 
const std::vector<T,A>& value, mpl::false_ primitive) const
{
this->send_impl(dest, tag, value, primitive);
}

template<typename T, typename A>
void communicator::send(int dest, int tag, const std::vector<T,A>& value) const
{
send_vector(dest, tag, value, is_mpi_datatype<T>());
}

template<typename T>
void communicator::send(int dest, int tag, const T* values, int n) const
{
this->array_send_impl(dest, tag, values, n, is_mpi_datatype<T>());
}

template<typename T>
status communicator::recv_impl(int source, int tag, T& value, mpl::true_) const
{
status stat;
BOOST_MPI_CHECK_RESULT(MPI_Recv,
(const_cast<T*>(&value), 1, 
get_mpi_datatype<T>(value),
source, tag, MPI_Comm(*this), &stat.m_status));
return stat;
}

template<typename T>
status
communicator::recv_impl(int source, int tag, T& value, mpl::false_) const
{
packed_iarchive ia(*this);
status stat = recv(source, tag, ia);

ia >> value;

return stat;
}

template<typename T>
status communicator::recv(int source, int tag, T& value) const
{
return this->recv_impl(source, tag, value, is_mpi_datatype<T>());
}

template<typename T>
status 
communicator::array_recv_impl(int source, int tag, T* values, int n, 
mpl::true_) const
{
status stat;
BOOST_MPI_CHECK_RESULT(MPI_Recv,
(const_cast<T*>(values), n, 
get_mpi_datatype<T>(*values),
source, tag, MPI_Comm(*this), &stat.m_status));
return stat;
}

template<typename T>
status
communicator::array_recv_impl(int source, int tag, T* values, int n, 
mpl::false_) const
{
packed_iarchive ia(*this);
status stat = recv(source, tag, ia);
T* v = values;
while (v != values+n) {
ia >> *v++;
}
stat.m_count = n;
return stat;
}

template<typename T, typename A>
status communicator::recv_vector(int source, int tag, 
std::vector<T,A>& values, mpl::true_ primitive) const
{
#if defined(BOOST_MPI_USE_IMPROBE)
{
MPI_Message msg;
status stat;
BOOST_MPI_CHECK_RESULT(MPI_Mprobe, (source,tag,*this,&msg,&stat.m_status));
int count;
BOOST_MPI_CHECK_RESULT(MPI_Get_count, (&stat.m_status,get_mpi_datatype<T>(),&count));
values.resize(count);
BOOST_MPI_CHECK_RESULT(MPI_Mrecv, (values.data(), count, get_mpi_datatype<T>(), &msg, &stat.m_status));
return stat;
}
#else
{
typename std::vector<T,A>::size_type size = 0;
recv(source, tag, size);
values.resize(size);
return this->array_recv_impl(source, tag, values.data(), size, primitive);
}
#endif
}

template<typename T, typename A>
status communicator::recv_vector(int source, int tag, 
std::vector<T,A>& value, mpl::false_ false_type) const
{
return this->recv_impl(source, tag, value, false_type);
}

template<typename T, typename A>
status communicator::recv(int source, int tag, std::vector<T,A>& value) const
{
return recv_vector(source, tag, value, is_mpi_datatype<T>());
}

template<typename T>
status communicator::recv(int source, int tag, T* values, int n) const
{
return this->array_recv_impl(source, tag, values, n, is_mpi_datatype<T>());
}


template<typename T>
status communicator::sendrecv_impl(int dest, int stag, const T& sval, int src, int rtag, T& rval,
mpl::true_) const
{
status stat;
BOOST_MPI_CHECK_RESULT(MPI_Sendrecv,
(const_cast<T*>(&sval), 1, 
get_mpi_datatype<T>(sval),
dest, stag, 
&rval, 1,
get_mpi_datatype<T>(rval),
src, rtag,
MPI_Comm(*this), &stat.m_status));
return stat;
}

template<typename T>
status communicator::sendrecv_impl(int dest, int stag, const T& sval, int src, int rtag, T& rval,
mpl::false_) const
{
int const SEND = 0;
int const RECV = 1;
request srrequests[2];
srrequests[SEND] = this->isend_impl(dest, stag, sval, mpl::false_());
srrequests[RECV] = this->irecv_impl(src,  rtag, rval, mpl::false_());
status srstatuses[2];
wait_all(srrequests, srrequests + 2, srstatuses);
return srstatuses[RECV];
}

template<typename T>
status communicator::sendrecv(int dest, int stag, const T& sval, int src, int rtag, T& rval) const
{
return this->sendrecv_impl(dest, stag, sval, src, rtag, rval, is_mpi_datatype<T>());
}


template<typename T>
request
communicator::isend_impl(int dest, int tag, const T& value, mpl::true_) const
{
return request::make_trivial_send(*this, dest, tag, value);
}

template<typename T>
request
communicator::isend_impl(int dest, int tag, const T& value, mpl::false_) const
{
shared_ptr<packed_oarchive> archive(new packed_oarchive(*this));
*archive << value;
request result = isend(dest, tag, *archive);
result.preserve(archive);
return result;
}

template<typename T>
request communicator::isend(int dest, int tag, const T& value) const
{
return this->isend_impl(dest, tag, value, is_mpi_datatype<T>());
}

template<typename T, class A>
request communicator::isend(int dest, int tag, const std::vector<T,A>& values) const
{
return this->isend_vector(dest, tag, values, is_mpi_datatype<T>());
}

template<typename T, class A>
request
communicator::isend_vector(int dest, int tag, const std::vector<T,A>& values,
mpl::true_ primitive) const
{
return request::make_dynamic_primitive_array_send(*this, dest, tag, values);
}

template<typename T, class A>
request
communicator::isend_vector(int dest, int tag, const std::vector<T,A>& values,
mpl::false_ no) const 
{
return this->isend_impl(dest, tag, values, no);
}

template<typename T>
request
communicator::array_isend_impl(int dest, int tag, const T* values, int n,
mpl::true_) const
{
return request::make_trivial_send(*this, dest, tag, values, n);
}

template<typename T>
request
communicator::array_isend_impl(int dest, int tag, const T* values, int n, 
mpl::false_) const
{
shared_ptr<packed_oarchive> archive(new packed_oarchive(*this));
T const* v = values;
while (v < values+n) {
*archive << *v++;
}
request result = isend(dest, tag, *archive);
result.preserve(archive);
return result;
}


template<typename T>
request communicator::isend(int dest, int tag, const T* values, int n) const
{
return array_isend_impl(dest, tag, values, n, is_mpi_datatype<T>());
}

template<typename T>
request 
communicator::irecv_impl(int source, int tag, T& value, mpl::true_) const
{
return request::make_trivial_recv(*this, source, tag, value);
}

template<typename T>
request
communicator::irecv_impl(int source, int tag, T& value, mpl::false_) const
{
return request::make_serialized(*this, source, tag, value);
}

template<typename T>
request 
communicator::irecv(int source, int tag, T& value) const
{
return this->irecv_impl(source, tag, value, is_mpi_datatype<T>());
}

template<typename T>
request 
communicator::array_irecv_impl(int source, int tag, T* values, int n, 
mpl::true_) const
{
return request::make_trivial_recv(*this, source, tag, values, n);
}

template<typename T>
request
communicator::array_irecv_impl(int source, int tag, T* values, int n, 
mpl::false_) const
{
return request::make_serialized_array(*this, source, tag, values, n);
}

template<typename T, class A>
request
communicator::irecv_vector(int source, int tag, std::vector<T,A>& values, 
mpl::true_ primitive) const
{
return request::make_dynamic_primitive_array_recv(*this, source, tag, values);
}

template<typename T, class A>
request
communicator::irecv_vector(int source, int tag, std::vector<T,A>& values, 
mpl::false_ no) const
{
return irecv_impl(source, tag, values, no);
}

template<typename T, typename A>
request
communicator::irecv(int source, int tag, std::vector<T,A>& values) const
{
return irecv_vector(source, tag, values, is_mpi_datatype<T>());
}

template<typename T>
request communicator::irecv(int source, int tag, T* values, int n) const
{
return this->array_irecv_impl(source, tag, values, n, is_mpi_datatype<T>());
}

} } 

#ifdef BOOST_MPI_SKELETON_AND_CONTENT_HPP
#  include <boost/mpi/detail/communicator_sc.hpp>
#endif

#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif

#endif 
