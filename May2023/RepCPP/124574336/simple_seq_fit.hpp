
#ifndef BOOST_INTERPROCESS_SIMPLE_SEQ_FIT_HPP
#define BOOST_INTERPROCESS_SIMPLE_SEQ_FIT_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/mem_algo/detail/simple_seq_fit_impl.hpp>
#include <boost/interprocess/offset_ptr.hpp>


namespace boost {
namespace interprocess {

template<class MutexFamily, class VoidPointer>
class simple_seq_fit
: public ipcdetail::simple_seq_fit_impl<MutexFamily, VoidPointer>
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
typedef ipcdetail::simple_seq_fit_impl<MutexFamily, VoidPointer> base_t;
#endif   

public:
typedef typename base_t::size_type                            size_type;

simple_seq_fit(size_type segment_size, size_type extra_hdr_bytes)
: base_t(segment_size, extra_hdr_bytes){}
};

}  

}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   

