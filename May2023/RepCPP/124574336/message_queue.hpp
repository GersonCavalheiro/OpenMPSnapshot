
#ifndef BOOST_INTERPROCESS_MESSAGE_QUEUE_HPP
#define BOOST_INTERPROCESS_MESSAGE_QUEUE_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/detail/managed_open_or_create_impl.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/permissions.hpp>
#include <boost/core/no_exceptions_support.hpp>
#include <boost/interprocess/detail/type_traits.hpp>
#include <boost/intrusive/pointer_traits.hpp>
#include <boost/move/detail/type_traits.hpp> 
#include <boost/intrusive/pointer_traits.hpp>
#include <boost/assert.hpp>
#include <algorithm> 
#include <cstddef>   
#include <cstring>   



namespace boost{  namespace interprocess{

namespace ipcdetail
{
template<class VoidPointer>
class msg_queue_initialization_func_t;
}

template<class VoidPointer>
class message_queue_t
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
enum block_t   {  blocking,   timed,   non_blocking   };

message_queue_t();
#endif   

public:
typedef VoidPointer                                                 void_pointer;
typedef typename boost::intrusive::
pointer_traits<void_pointer>::template
rebind_pointer<char>::type                                    char_ptr;
typedef typename boost::intrusive::pointer_traits<char_ptr>::difference_type difference_type;
typedef typename boost::container::dtl::make_unsigned<difference_type>::type        size_type;

message_queue_t(create_only_t create_only,
const char *name,
size_type max_num_msg,
size_type max_msg_size,
const permissions &perm = permissions());

message_queue_t(open_or_create_t open_or_create,
const char *name,
size_type max_num_msg,
size_type max_msg_size,
const permissions &perm = permissions());

message_queue_t(open_only_t open_only,
const char *name);

~message_queue_t();

void send (const void *buffer,     size_type buffer_size,
unsigned int priority);

bool try_send    (const void *buffer,     size_type buffer_size,
unsigned int priority);

bool timed_send    (const void *buffer,     size_type buffer_size,
unsigned int priority,  const boost::posix_time::ptime& abs_time);

void receive (void *buffer,           size_type buffer_size,
size_type &recvd_size,unsigned int &priority);

bool try_receive (void *buffer,           size_type buffer_size,
size_type &recvd_size,unsigned int &priority);

bool timed_receive (void *buffer,           size_type buffer_size,
size_type &recvd_size,unsigned int &priority,
const boost::posix_time::ptime &abs_time);

size_type get_max_msg() const;

size_type get_max_msg_size() const;

size_type get_num_msg() const;

static bool remove(const char *name);

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
typedef boost::posix_time::ptime ptime;

friend class ipcdetail::msg_queue_initialization_func_t<VoidPointer>;

bool do_receive(block_t block,
void *buffer,         size_type buffer_size,
size_type &recvd_size, unsigned int &priority,
const ptime &abs_time);

bool do_send(block_t block,
const void *buffer,      size_type buffer_size,
unsigned int priority,   const ptime &abs_time);

static size_type get_mem_size(size_type max_msg_size, size_type max_num_msg);
typedef ipcdetail::managed_open_or_create_impl<shared_memory_object, 0, true, false> open_create_impl_t;
open_create_impl_t m_shmem;
#endif   
};

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

namespace ipcdetail {

template<class VoidPointer>
class msg_hdr_t
{
typedef VoidPointer                                                           void_pointer;
typedef typename boost::intrusive::
pointer_traits<void_pointer>::template
rebind_pointer<char>::type                                              char_ptr;
typedef typename boost::intrusive::pointer_traits<char_ptr>::difference_type  difference_type;
typedef typename boost::container::dtl::make_unsigned<difference_type>::type                  size_type;

public:
size_type               len;     
unsigned int            priority;
void * data(){ return this+1; }  
};

template<class VoidPointer>
class priority_functor
{
typedef typename boost::intrusive::
pointer_traits<VoidPointer>::template
rebind_pointer<msg_hdr_t<VoidPointer> >::type                  msg_hdr_ptr_t;

public:
bool operator()(const msg_hdr_ptr_t &msg1,
const msg_hdr_ptr_t &msg2) const
{  return msg1->priority < msg2->priority;  }
};

template<class VoidPointer>
class mq_hdr_t
: public ipcdetail::priority_functor<VoidPointer>
{
typedef VoidPointer                                                     void_pointer;
typedef msg_hdr_t<void_pointer>                                         msg_header;
typedef typename boost::intrusive::
pointer_traits<void_pointer>::template
rebind_pointer<msg_header>::type                                  msg_hdr_ptr_t;
typedef typename boost::intrusive::pointer_traits
<msg_hdr_ptr_t>::difference_type                                     difference_type;
typedef typename boost::container::
dtl::make_unsigned<difference_type>::type               size_type;
typedef typename boost::intrusive::
pointer_traits<void_pointer>::template
rebind_pointer<msg_hdr_ptr_t>::type                              msg_hdr_ptr_ptr_t;
typedef ipcdetail::managed_open_or_create_impl<shared_memory_object, 0, true, false> open_create_impl_t;

public:
mq_hdr_t(size_type max_num_msg, size_type max_msg_size)
: m_max_num_msg(max_num_msg),
m_max_msg_size(max_msg_size),
m_cur_num_msg(0)
#if defined(BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX)
,m_cur_first_msg(0u)
,m_blocked_senders(0u)
,m_blocked_receivers(0u)
#endif
{  this->initialize_memory();  }

bool is_full() const
{  return m_cur_num_msg == m_max_num_msg;  }

bool is_empty() const
{  return !m_cur_num_msg;  }

void free_top_msg()
{  --m_cur_num_msg;  }

#if defined(BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX)

typedef msg_hdr_ptr_t *iterator;

size_type end_pos() const
{
const size_type space_until_bufend = m_max_num_msg - m_cur_first_msg;
return space_until_bufend > m_cur_num_msg
? m_cur_first_msg + m_cur_num_msg : m_cur_num_msg - space_until_bufend;
}

msg_header &top_msg()
{
size_type pos = this->end_pos();
return *mp_index[pos ? --pos : m_max_num_msg - 1];
}

msg_header &bottom_msg()
{  return *mp_index[m_cur_first_msg];   }

iterator inserted_ptr_begin() const
{  return &mp_index[m_cur_first_msg]; }

iterator inserted_ptr_end() const
{  return &mp_index[this->end_pos()];  }

iterator lower_bound(const msg_hdr_ptr_t & value, priority_functor<VoidPointer> func)
{
iterator begin(this->inserted_ptr_begin()), end(this->inserted_ptr_end());
if(end < begin){
iterator idx_end = &mp_index[m_max_num_msg];
iterator ret = std::lower_bound(begin, idx_end, value, func);
if(idx_end == ret){
iterator idx_beg = &mp_index[0];
ret = std::lower_bound(idx_beg, end, value, func);
BOOST_ASSERT(ret != end);
BOOST_ASSERT(ret != begin);
return ret;
}
else{
return ret;
}
}
else{
return std::lower_bound(begin, end, value, func);
}
}

msg_header & insert_at(iterator where)
{
iterator it_inserted_ptr_end = this->inserted_ptr_end();
iterator it_inserted_ptr_beg = this->inserted_ptr_begin();
if(where == it_inserted_ptr_beg){
m_cur_first_msg = m_cur_first_msg ? m_cur_first_msg : m_max_num_msg;
--m_cur_first_msg;
++m_cur_num_msg;
return *mp_index[m_cur_first_msg];
}
else if(where == it_inserted_ptr_end){
++m_cur_num_msg;
return **it_inserted_ptr_end;
}
else{
size_type pos  = where - &mp_index[0];
size_type circ_pos = pos >= m_cur_first_msg ? pos - m_cur_first_msg : pos + (m_max_num_msg - m_cur_first_msg);
if(circ_pos < m_cur_num_msg/2){
if(!pos){
pos   = m_max_num_msg;
where = &mp_index[m_max_num_msg-1];
}
else{
--where;
}
const bool unique_segment = m_cur_first_msg && m_cur_first_msg <= pos;
const size_type first_segment_beg  = unique_segment ? m_cur_first_msg : 1u;
const size_type first_segment_end  = pos;
const size_type second_segment_beg = unique_segment || !m_cur_first_msg ? m_max_num_msg : m_cur_first_msg;
const size_type second_segment_end = m_max_num_msg;
const msg_hdr_ptr_t backup   = *(&mp_index[0] + (unique_segment ?  first_segment_beg : second_segment_beg) - 1);

if(!unique_segment){
std::copy( &mp_index[0] + second_segment_beg
, &mp_index[0] + second_segment_end
, &mp_index[0] + second_segment_beg - 1);
mp_index[m_max_num_msg-1] = mp_index[0];
}
std::copy( &mp_index[0] + first_segment_beg
, &mp_index[0] + first_segment_end
, &mp_index[0] + first_segment_beg - 1);
*where = backup;
m_cur_first_msg = m_cur_first_msg ? m_cur_first_msg : m_max_num_msg;
--m_cur_first_msg;
++m_cur_num_msg;
return **where;
}
else{
const size_type pos_end = this->end_pos();
const bool unique_segment = pos < pos_end;
const size_type first_segment_beg  = pos;
const size_type first_segment_end  = unique_segment  ? pos_end : m_max_num_msg-1;
const size_type second_segment_beg = 0u;
const size_type second_segment_end = unique_segment ? 0u : pos_end;
const msg_hdr_ptr_t backup   = *it_inserted_ptr_end;

if(!unique_segment){
std::copy_backward( &mp_index[0] + second_segment_beg
, &mp_index[0] + second_segment_end
, &mp_index[0] + second_segment_end + 1);
mp_index[0] = mp_index[m_max_num_msg-1];
}
std::copy_backward( &mp_index[0] + first_segment_beg
, &mp_index[0] + first_segment_end
, &mp_index[0] + first_segment_end + 1);
*where = backup;
++m_cur_num_msg;
return **where;
}
}
}

#else 

typedef msg_hdr_ptr_t *iterator;

msg_header &top_msg()
{  return *mp_index[m_cur_num_msg-1];   }

msg_header &bottom_msg()
{  return *mp_index[0];   }

iterator inserted_ptr_begin() const
{  return &mp_index[0]; }

iterator inserted_ptr_end() const
{  return &mp_index[m_cur_num_msg]; }

iterator lower_bound(const msg_hdr_ptr_t & value, priority_functor<VoidPointer> func)
{  return std::lower_bound(this->inserted_ptr_begin(), this->inserted_ptr_end(), value, func);  }

msg_header & insert_at(iterator pos)
{
const msg_hdr_ptr_t backup = *inserted_ptr_end();
std::copy_backward(pos, inserted_ptr_end(), inserted_ptr_end()+1);
*pos = backup;
++m_cur_num_msg;
return **pos;
}

#endif   

msg_header & queue_free_msg(unsigned int priority)
{
iterator it  (inserted_ptr_begin()), it_end(inserted_ptr_end());
if(m_cur_num_msg && priority > this->bottom_msg().priority){
if(priority > this->top_msg().priority){
it = it_end;
}
else{
msg_header dummy_hdr;
dummy_hdr.priority = priority;

msg_hdr_ptr_t dummy_ptr(&dummy_hdr);

it = this->lower_bound(dummy_ptr, static_cast<priority_functor<VoidPointer>&>(*this));
}
}
return this->insert_at(it);
}

static size_type get_mem_size
(size_type max_msg_size, size_type max_num_msg)
{
const size_type
msg_hdr_align  = ::boost::container::dtl::alignment_of<msg_header>::value,
index_align    = ::boost::container::dtl::alignment_of<msg_hdr_ptr_t>::value,
r_hdr_size     = ipcdetail::ct_rounded_size<sizeof(mq_hdr_t), index_align>::value,
r_index_size   = ipcdetail::get_rounded_size<size_type>(max_num_msg*sizeof(msg_hdr_ptr_t), msg_hdr_align),
r_max_msg_size = ipcdetail::get_rounded_size<size_type>(max_msg_size, msg_hdr_align) + sizeof(msg_header);
return r_hdr_size + r_index_size + (max_num_msg*r_max_msg_size) +
open_create_impl_t::ManagedOpenOrCreateUserOffset;
}

void initialize_memory()
{
const size_type
msg_hdr_align  = ::boost::container::dtl::alignment_of<msg_header>::value,
index_align    = ::boost::container::dtl::alignment_of<msg_hdr_ptr_t>::value,
r_hdr_size     = ipcdetail::ct_rounded_size<sizeof(mq_hdr_t), index_align>::value,
r_index_size   = ipcdetail::get_rounded_size<size_type>(m_max_num_msg*sizeof(msg_hdr_ptr_t), msg_hdr_align),
r_max_msg_size = ipcdetail::get_rounded_size<size_type>(m_max_msg_size, msg_hdr_align) + sizeof(msg_header);

msg_hdr_ptr_t *index =  reinterpret_cast<msg_hdr_ptr_t*>
(reinterpret_cast<char*>(this)+r_hdr_size);

msg_header *msg_hdr   =  reinterpret_cast<msg_header*>
(reinterpret_cast<char*>(this)+r_hdr_size+r_index_size);

mp_index             = index;

for(size_type i = 0; i < m_max_num_msg; ++i){
index[i] = msg_hdr;
msg_hdr  = reinterpret_cast<msg_header*>
(reinterpret_cast<char*>(msg_hdr)+r_max_msg_size);
}
}

public:
msg_hdr_ptr_ptr_t          mp_index;
const size_type            m_max_num_msg;
const size_type            m_max_msg_size;
size_type                  m_cur_num_msg;
interprocess_mutex         m_mutex;
interprocess_condition     m_cond_recv;
interprocess_condition     m_cond_send;
#if defined(BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX)
size_type                  m_cur_first_msg;
size_type                  m_blocked_senders;
size_type                  m_blocked_receivers;
#endif
};


template<class VoidPointer>
class msg_queue_initialization_func_t
{
public:
typedef typename boost::intrusive::
pointer_traits<VoidPointer>::template
rebind_pointer<char>::type                               char_ptr;
typedef typename boost::intrusive::pointer_traits<char_ptr>::
difference_type                                             difference_type;
typedef typename boost::container::dtl::
make_unsigned<difference_type>::type                        size_type;

msg_queue_initialization_func_t(size_type maxmsg = 0,
size_type maxmsgsize = 0)
: m_maxmsg (maxmsg), m_maxmsgsize(maxmsgsize) {}

bool operator()(void *address, size_type, bool created)
{
char      *mptr;

if(created){
mptr     = reinterpret_cast<char*>(address);
BOOST_TRY{
new (mptr) mq_hdr_t<VoidPointer>(m_maxmsg, m_maxmsgsize);
}
BOOST_CATCH(...){
return false;
}
BOOST_CATCH_END
}
return true;
}

std::size_t get_min_size() const
{
return mq_hdr_t<VoidPointer>::get_mem_size(m_maxmsgsize, m_maxmsg)
- message_queue_t<VoidPointer>::open_create_impl_t::ManagedOpenOrCreateUserOffset;
}

const size_type m_maxmsg;
const size_type m_maxmsgsize;
};

}  

template<class VoidPointer>
inline message_queue_t<VoidPointer>::~message_queue_t()
{}

template<class VoidPointer>
inline typename message_queue_t<VoidPointer>::size_type message_queue_t<VoidPointer>::get_mem_size
(size_type max_msg_size, size_type max_num_msg)
{  return ipcdetail::mq_hdr_t<VoidPointer>::get_mem_size(max_msg_size, max_num_msg);   }

template<class VoidPointer>
inline message_queue_t<VoidPointer>::message_queue_t(create_only_t,
const char *name,
size_type max_num_msg,
size_type max_msg_size,
const permissions &perm)
:  m_shmem(create_only,
name,
get_mem_size(max_msg_size, max_num_msg),
read_write,
static_cast<void*>(0),
ipcdetail::msg_queue_initialization_func_t<VoidPointer> (max_num_msg, max_msg_size),
perm)
{}

template<class VoidPointer>
inline message_queue_t<VoidPointer>::message_queue_t(open_or_create_t,
const char *name,
size_type max_num_msg,
size_type max_msg_size,
const permissions &perm)
:  m_shmem(open_or_create,
name,
get_mem_size(max_msg_size, max_num_msg),
read_write,
static_cast<void*>(0),
ipcdetail::msg_queue_initialization_func_t<VoidPointer> (max_num_msg, max_msg_size),
perm)
{}

template<class VoidPointer>
inline message_queue_t<VoidPointer>::message_queue_t(open_only_t, const char *name)
:  m_shmem(open_only,
name,
read_write,
static_cast<void*>(0),
ipcdetail::msg_queue_initialization_func_t<VoidPointer> ())
{}

template<class VoidPointer>
inline void message_queue_t<VoidPointer>::send
(const void *buffer, size_type buffer_size, unsigned int priority)
{  this->do_send(blocking, buffer, buffer_size, priority, ptime()); }

template<class VoidPointer>
inline bool message_queue_t<VoidPointer>::try_send
(const void *buffer, size_type buffer_size, unsigned int priority)
{  return this->do_send(non_blocking, buffer, buffer_size, priority, ptime()); }

template<class VoidPointer>
inline bool message_queue_t<VoidPointer>::timed_send
(const void *buffer, size_type buffer_size
,unsigned int priority, const boost::posix_time::ptime &abs_time)
{
if(abs_time.is_pos_infinity()){
this->send(buffer, buffer_size, priority);
return true;
}
return this->do_send(timed, buffer, buffer_size, priority, abs_time);
}

template<class VoidPointer>
inline bool message_queue_t<VoidPointer>::do_send(block_t block,
const void *buffer,      size_type buffer_size,
unsigned int priority,   const boost::posix_time::ptime &abs_time)
{
ipcdetail::mq_hdr_t<VoidPointer> *p_hdr = static_cast<ipcdetail::mq_hdr_t<VoidPointer>*>(m_shmem.get_user_address());
if (buffer_size > p_hdr->m_max_msg_size) {
throw interprocess_exception(size_error);
}

#if defined(BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX)
bool notify_blocked_receivers = false;
#endif
scoped_lock<interprocess_mutex> lock(p_hdr->m_mutex);
{
if (p_hdr->is_full()) {
BOOST_TRY{
#ifdef BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX
++p_hdr->m_blocked_senders;
#endif
switch(block){
case non_blocking :
#ifdef BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX
--p_hdr->m_blocked_senders;
#endif
return false;
break;

case blocking :
do{
p_hdr->m_cond_send.wait(lock);
}
while (p_hdr->is_full());
break;

case timed :
do{
if(!p_hdr->m_cond_send.timed_wait(lock, abs_time)){
if(p_hdr->is_full()){
#ifdef BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX
--p_hdr->m_blocked_senders;
#endif
return false;
}
break;
}
}
while (p_hdr->is_full());
break;
default:
break;
}
#ifdef BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX
--p_hdr->m_blocked_senders;
#endif
}
BOOST_CATCH(...){
#ifdef BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX
--p_hdr->m_blocked_senders;
#endif
BOOST_RETHROW;
}
BOOST_CATCH_END
}

#if defined(BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX)
notify_blocked_receivers = 0 != p_hdr->m_blocked_receivers;
#endif
ipcdetail::msg_hdr_t<VoidPointer> &free_msg_hdr = p_hdr->queue_free_msg(priority);

BOOST_ASSERT(free_msg_hdr.priority == 0);
BOOST_ASSERT(free_msg_hdr.len == 0);

free_msg_hdr.priority = priority;
free_msg_hdr.len      = buffer_size;

std::memcpy(free_msg_hdr.data(), buffer, buffer_size);
}  

#if defined(BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX)
if (notify_blocked_receivers){
p_hdr->m_cond_recv.notify_one();
}
#else
p_hdr->m_cond_recv.notify_one();
#endif

return true;
}

template<class VoidPointer>
inline void message_queue_t<VoidPointer>::receive(void *buffer,        size_type buffer_size,
size_type &recvd_size,   unsigned int &priority)
{  this->do_receive(blocking, buffer, buffer_size, recvd_size, priority, ptime()); }

template<class VoidPointer>
inline bool
message_queue_t<VoidPointer>::try_receive(void *buffer,              size_type buffer_size,
size_type &recvd_size,   unsigned int &priority)
{  return this->do_receive(non_blocking, buffer, buffer_size, recvd_size, priority, ptime()); }

template<class VoidPointer>
inline bool
message_queue_t<VoidPointer>::timed_receive(void *buffer,            size_type buffer_size,
size_type &recvd_size,   unsigned int &priority,
const boost::posix_time::ptime &abs_time)
{
if(abs_time.is_pos_infinity()){
this->receive(buffer, buffer_size, recvd_size, priority);
return true;
}
return this->do_receive(timed, buffer, buffer_size, recvd_size, priority, abs_time);
}

template<class VoidPointer>
inline bool
message_queue_t<VoidPointer>::do_receive(block_t block,
void *buffer,            size_type buffer_size,
size_type &recvd_size,   unsigned int &priority,
const boost::posix_time::ptime &abs_time)
{
ipcdetail::mq_hdr_t<VoidPointer> *p_hdr = static_cast<ipcdetail::mq_hdr_t<VoidPointer>*>(m_shmem.get_user_address());
if (buffer_size < p_hdr->m_max_msg_size) {
throw interprocess_exception(size_error);
}

#if defined(BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX)
bool notify_blocked_senders = false;
#endif
scoped_lock<interprocess_mutex> lock(p_hdr->m_mutex);
{
if (p_hdr->is_empty()) {
BOOST_TRY{
#if defined(BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX)
++p_hdr->m_blocked_receivers;
#endif
switch(block){
case non_blocking :
#if defined(BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX)
--p_hdr->m_blocked_receivers;
#endif
return false;
break;

case blocking :
do{
p_hdr->m_cond_recv.wait(lock);
}
while (p_hdr->is_empty());
break;

case timed :
do{
if(!p_hdr->m_cond_recv.timed_wait(lock, abs_time)){
if(p_hdr->is_empty()){
#if defined(BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX)
--p_hdr->m_blocked_receivers;
#endif
return false;
}
break;
}
}
while (p_hdr->is_empty());
break;

default:
break;
}
#if defined(BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX)
--p_hdr->m_blocked_receivers;
#endif
}
BOOST_CATCH(...){
#if defined(BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX)
--p_hdr->m_blocked_receivers;
#endif
BOOST_RETHROW;
}
BOOST_CATCH_END
}

#ifdef BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX
notify_blocked_senders = 0 != p_hdr->m_blocked_senders;
#endif

ipcdetail::msg_hdr_t<VoidPointer> &top_msg = p_hdr->top_msg();

recvd_size     = top_msg.len;
priority       = top_msg.priority;

top_msg.len       = 0;
top_msg.priority  = 0;

std::memcpy(buffer, top_msg.data(), recvd_size);

p_hdr->free_top_msg();
}  

#ifdef BOOST_INTERPROCESS_MSG_QUEUE_CIRCULAR_INDEX
if (notify_blocked_senders){
p_hdr->m_cond_send.notify_one();
}
#else
p_hdr->m_cond_send.notify_one();
#endif

return true;
}

template<class VoidPointer>
inline typename message_queue_t<VoidPointer>::size_type message_queue_t<VoidPointer>::get_max_msg() const
{
ipcdetail::mq_hdr_t<VoidPointer> *p_hdr = static_cast<ipcdetail::mq_hdr_t<VoidPointer>*>(m_shmem.get_user_address());
return p_hdr ? p_hdr->m_max_num_msg : 0;  }

template<class VoidPointer>
inline typename message_queue_t<VoidPointer>::size_type message_queue_t<VoidPointer>::get_max_msg_size() const
{
ipcdetail::mq_hdr_t<VoidPointer> *p_hdr = static_cast<ipcdetail::mq_hdr_t<VoidPointer>*>(m_shmem.get_user_address());
return p_hdr ? p_hdr->m_max_msg_size : 0;
}

template<class VoidPointer>
inline typename message_queue_t<VoidPointer>::size_type message_queue_t<VoidPointer>::get_num_msg() const
{
ipcdetail::mq_hdr_t<VoidPointer> *p_hdr = static_cast<ipcdetail::mq_hdr_t<VoidPointer>*>(m_shmem.get_user_address());
if(p_hdr){
scoped_lock<interprocess_mutex> lock(p_hdr->m_mutex);
return p_hdr->m_cur_num_msg;
}

return 0;
}

template<class VoidPointer>
inline bool message_queue_t<VoidPointer>::remove(const char *name)
{  return shared_memory_object::remove(name);  }

#else

typedef message_queue_t<offset_ptr<void> > message_queue;

#endif   

}} 

#include <boost/interprocess/detail/config_end.hpp>

#endif   
