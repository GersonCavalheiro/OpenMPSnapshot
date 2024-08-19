

#include "internal/_deprecated_header_message_guard.h"

#if !defined(__TBB_show_deprecation_message_reader_writer_lock_H) && defined(__TBB_show_deprecated_header_message)
#define  __TBB_show_deprecation_message_reader_writer_lock_H
#pragma message("TBB Warning: tbb/reader_writer_lock.h is deprecated. For details, please see Deprecated Features appendix in the TBB reference manual.")
#endif

#if defined(__TBB_show_deprecated_header_message)
#undef __TBB_show_deprecated_header_message
#endif

#ifndef __TBB_reader_writer_lock_H
#define __TBB_reader_writer_lock_H

#define __TBB_reader_writer_lock_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "tbb_thread.h"
#include "tbb_allocator.h"
#include "atomic.h"

namespace tbb {
namespace interface5 {

class __TBB_DEPRECATED_VERBOSE_MSG("tbb::reader_writer_lock is deprecated, use std::shared_mutex")
reader_writer_lock : tbb::internal::no_copy {
public:
friend class scoped_lock;
friend class scoped_lock_read;

enum status_t { waiting_nonblocking, waiting, active, invalid };

reader_writer_lock() {
internal_construct();
}

~reader_writer_lock() {
internal_destroy();
}


class scoped_lock : tbb::internal::no_copy {
public:
friend class reader_writer_lock;

scoped_lock(reader_writer_lock& lock) {
internal_construct(lock);
}

~scoped_lock() {
internal_destroy();
}

void* operator new(size_t s) {
return tbb::internal::allocate_via_handler_v3(s);
}
void operator delete(void* p) {
tbb::internal::deallocate_via_handler_v3(p);
}

private:
reader_writer_lock *mutex;
scoped_lock* next;
atomic<status_t> status;

scoped_lock();

void __TBB_EXPORTED_METHOD internal_construct(reader_writer_lock&);
void __TBB_EXPORTED_METHOD internal_destroy();
};

class scoped_lock_read : tbb::internal::no_copy {
public:
friend class reader_writer_lock;

scoped_lock_read(reader_writer_lock& lock) {
internal_construct(lock);
}

~scoped_lock_read() {
internal_destroy();
}

void* operator new(size_t s) {
return tbb::internal::allocate_via_handler_v3(s);
}
void operator delete(void* p) {
tbb::internal::deallocate_via_handler_v3(p);
}

private:
reader_writer_lock *mutex;
scoped_lock_read *next;
atomic<status_t> status;

scoped_lock_read();

void __TBB_EXPORTED_METHOD internal_construct(reader_writer_lock&);
void __TBB_EXPORTED_METHOD internal_destroy();
};


void __TBB_EXPORTED_METHOD lock();


bool __TBB_EXPORTED_METHOD try_lock();


void __TBB_EXPORTED_METHOD lock_read();


bool __TBB_EXPORTED_METHOD try_lock_read();

void __TBB_EXPORTED_METHOD unlock();

private:
void __TBB_EXPORTED_METHOD internal_construct();
void __TBB_EXPORTED_METHOD internal_destroy();


bool start_write(scoped_lock *);
void set_next_writer(scoped_lock *w);
void end_write(scoped_lock *);
bool is_current_writer();


void start_read(scoped_lock_read *);
void unblock_readers();
void end_read();

atomic<scoped_lock_read*> reader_head;
atomic<scoped_lock*> writer_head;
atomic<scoped_lock*> writer_tail;
tbb_thread::id my_current_writer;
atomic<uintptr_t> rdr_count_and_flags; 
};

} 

using interface5::reader_writer_lock;

} 

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_reader_writer_lock_H_include_area

#endif 
