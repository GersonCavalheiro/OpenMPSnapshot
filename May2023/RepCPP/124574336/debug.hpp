


#if !defined(BOOST_CIRCULAR_BUFFER_DEBUG_HPP)
#define BOOST_CIRCULAR_BUFFER_DEBUG_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#if BOOST_CB_ENABLE_DEBUG
#include <cstring>

#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std {
using ::memset;
}
#endif

#endif 
namespace boost {

namespace cb_details {

#if BOOST_CB_ENABLE_DEBUG

const int UNINITIALIZED = 0xcc;

template <class T>
inline void do_fill_uninitialized_memory(T* data, std::size_t size_in_bytes) BOOST_NOEXCEPT {
std::memset(static_cast<void*>(data), UNINITIALIZED, size_in_bytes);
}

template <class T>
inline void do_fill_uninitialized_memory(T& , std::size_t ) BOOST_NOEXCEPT {
}


class debug_iterator_registry;


class debug_iterator_base {

private:

mutable const debug_iterator_registry* m_registry;

mutable const debug_iterator_base* m_next;

public:

debug_iterator_base();

debug_iterator_base(const debug_iterator_registry* registry);

debug_iterator_base(const debug_iterator_base& rhs);

~debug_iterator_base();


debug_iterator_base& operator = (const debug_iterator_base& rhs);

bool is_valid(const debug_iterator_registry* registry) const;


void invalidate() const;

const debug_iterator_base* next() const;


void set_next(const debug_iterator_base* it) const;

private:

void register_self();

void unregister_self();
};


class debug_iterator_registry {

mutable const debug_iterator_base* m_iterators;

public:

debug_iterator_registry() : m_iterators(0) {}


void register_iterator(const debug_iterator_base* it) const {
it->set_next(m_iterators);
m_iterators = it;
}


void unregister_iterator(const debug_iterator_base* it) const {
const debug_iterator_base* previous = 0;
for (const debug_iterator_base* p = m_iterators; p != it; previous = p, p = p->next()) {}
remove(it, previous);
}

template <class Iterator>
void invalidate_iterators(const Iterator& it) {
const debug_iterator_base* previous = 0;
for (const debug_iterator_base* p = m_iterators; p != 0; p = p->next()) {
if (((Iterator*)p)->m_it == it.m_it) {
p->invalidate();
remove(p, previous);
continue;
}
previous = p;
}
}

template <class Iterator>
void invalidate_iterators_except(const Iterator& it) {
const debug_iterator_base* previous = 0;
for (const debug_iterator_base* p = m_iterators; p != 0; p = p->next()) {
if (((Iterator*)p)->m_it != it.m_it) {
p->invalidate();
remove(p, previous);
continue;
}
previous = p;
}
}

void invalidate_all_iterators() {
for (const debug_iterator_base* p = m_iterators; p != 0; p = p->next())
p->invalidate();
m_iterators = 0;
}

private:

void remove(const debug_iterator_base* current,
const debug_iterator_base* previous) const {
if (previous == 0)
m_iterators = m_iterators->next();
else
previous->set_next(current->next());
}
};


inline debug_iterator_base::debug_iterator_base() : m_registry(0), m_next(0) {}

inline debug_iterator_base::debug_iterator_base(const debug_iterator_registry* registry)
: m_registry(registry), m_next(0) {
register_self();
}

inline debug_iterator_base::debug_iterator_base(const debug_iterator_base& rhs)
: m_registry(rhs.m_registry), m_next(0) {
register_self();
}

inline debug_iterator_base::~debug_iterator_base() { unregister_self(); }

inline debug_iterator_base& debug_iterator_base::operator = (const debug_iterator_base& rhs) {
if (m_registry == rhs.m_registry)
return *this;
unregister_self();
m_registry = rhs.m_registry;
register_self();
return *this;
}

inline bool debug_iterator_base::is_valid(const debug_iterator_registry* registry) const {
return m_registry == registry;
}

inline void debug_iterator_base::invalidate() const { m_registry = 0; }

inline const debug_iterator_base* debug_iterator_base::next() const { return m_next; }

inline void debug_iterator_base::set_next(const debug_iterator_base* it) const { m_next = it; }

inline void debug_iterator_base::register_self() {
if (m_registry != 0)
m_registry->register_iterator(this);
}

inline void debug_iterator_base::unregister_self() {
if (m_registry != 0)
m_registry->unregister_iterator(this);
}

#endif 

} 

} 

#endif 
