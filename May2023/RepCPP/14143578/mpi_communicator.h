#pragma once

#include <mpi.h>

namespace dg
{
template<class value_type>
static inline MPI_Datatype getMPIDataType(){ assert( false && "Type not supported!\n" ); return MPI_DOUBLE; }
template<>
inline MPI_Datatype getMPIDataType<double>(){ return MPI_DOUBLE;}
template<>
inline MPI_Datatype getMPIDataType<float>(){ return MPI_FLOAT;}
template<>
inline MPI_Datatype getMPIDataType<int>(){ return MPI_INT;}
template<>
inline MPI_Datatype getMPIDataType<bool>(){ return MPI_C_BOOL;}
template<>
inline MPI_Datatype getMPIDataType<unsigned>(){ return MPI_UNSIGNED;}


template< class LocalContainer>
struct aCommunicator
{
using value_type = get_value_type<LocalContainer>;

using container_type = LocalContainer; 


LocalContainer allocate_buffer( )const{
if( do_size() == 0 ) return LocalContainer();
return do_make_buffer();
}


void global_gather( const value_type* values, LocalContainer& buffer)const
{
do_global_gather( values, buffer);
}


LocalContainer global_gather( const value_type* values) const
{
LocalContainer tmp = do_make_buffer();
do_global_gather( values, tmp);
return tmp;
}


void global_scatter_reduce( const LocalContainer& toScatter, value_type* values) const{
do_global_scatter_reduce(toScatter, values);
}


unsigned buffer_size() const{return do_size();}

unsigned local_size() const{return m_source_size;}

bool isCommunicating() const{
return do_isCommunicating();
}

MPI_Comm communicator() const{return do_communicator();}
virtual aCommunicator* clone() const =0;
virtual ~aCommunicator(){}
protected:
aCommunicator(unsigned local_size=0):m_source_size(local_size){}
aCommunicator(const aCommunicator& src):m_source_size( src.m_source_size){ }
aCommunicator& operator=(const aCommunicator& src){
m_source_size = src.m_source_size;
return *this;
}
void set_local_size( unsigned new_size){
m_source_size = new_size;
}
private:
unsigned m_source_size;
virtual MPI_Comm do_communicator() const=0;
virtual unsigned do_size() const=0;
virtual LocalContainer do_make_buffer( )const=0;
virtual void do_global_gather( const value_type* values, LocalContainer& gathered)const=0;
virtual void do_global_scatter_reduce( const LocalContainer& toScatter, value_type* values) const=0;
virtual bool do_isCommunicating( ) const {
return true;
}
};





}
