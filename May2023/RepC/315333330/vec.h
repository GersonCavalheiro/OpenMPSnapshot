#ifndef GCC_VEC_H
#define GCC_VEC_H
extern void ggc_free (void *);
extern size_t ggc_round_alloc_size (size_t requested_size);
extern void *ggc_realloc (void *, size_t MEM_STAT_DECL);
extern void dump_vec_loc_statistics (void);
extern htab_t vec_mem_usage_hash;
struct vec_prefix
{
void register_overhead (void *, size_t, size_t CXX_MEM_STAT_INFO);
void release_overhead (void *, size_t, bool CXX_MEM_STAT_INFO);
static unsigned calculate_allocation (vec_prefix *, unsigned, bool);
static unsigned calculate_allocation_1 (unsigned, unsigned);
template <typename, typename, typename> friend struct vec;
friend struct va_gc;
friend struct va_gc_atomic;
friend struct va_heap;
unsigned m_alloc : 31;
unsigned m_using_auto_storage : 1;
unsigned m_num;
};
inline unsigned
vec_prefix::calculate_allocation (vec_prefix *pfx, unsigned reserve,
bool exact)
{
if (exact)
return (pfx ? pfx->m_num : 0) + reserve;
else if (!pfx)
return MAX (4, reserve);
return calculate_allocation_1 (pfx->m_alloc, pfx->m_num + reserve);
}
template<typename, typename, typename> struct vec;
struct vl_embed { };
struct vl_ptr { };
struct va_heap
{
typedef vl_ptr default_layout;
template<typename T>
static void reserve (vec<T, va_heap, vl_embed> *&, unsigned, bool
CXX_MEM_STAT_INFO);
template<typename T>
static void release (vec<T, va_heap, vl_embed> *&);
};
template<typename T>
inline void
va_heap::reserve (vec<T, va_heap, vl_embed> *&v, unsigned reserve, bool exact
MEM_STAT_DECL)
{
unsigned alloc
= vec_prefix::calculate_allocation (v ? &v->m_vecpfx : 0, reserve, exact);
gcc_checking_assert (alloc);
if (GATHER_STATISTICS && v)
v->m_vecpfx.release_overhead (v, v->allocated (), false);
size_t size = vec<T, va_heap, vl_embed>::embedded_size (alloc);
unsigned nelem = v ? v->length () : 0;
v = static_cast <vec<T, va_heap, vl_embed> *> (xrealloc (v, size));
v->embedded_init (alloc, nelem);
if (GATHER_STATISTICS)
v->m_vecpfx.register_overhead (v, alloc, nelem PASS_MEM_STAT);
}
template<typename T>
void
va_heap::release (vec<T, va_heap, vl_embed> *&v)
{
if (v == NULL)
return;
if (GATHER_STATISTICS)
v->m_vecpfx.release_overhead (v, v->allocated (), true);
::free (v);
v = NULL;
}
struct va_gc
{
typedef vl_embed default_layout;
template<typename T, typename A>
static void reserve (vec<T, A, vl_embed> *&, unsigned, bool
CXX_MEM_STAT_INFO);
template<typename T, typename A>
static void release (vec<T, A, vl_embed> *&v);
};
template<typename T, typename A>
inline void
va_gc::release (vec<T, A, vl_embed> *&v)
{
if (v)
::ggc_free (v);
v = NULL;
}
template<typename T, typename A>
void
va_gc::reserve (vec<T, A, vl_embed> *&v, unsigned reserve, bool exact
MEM_STAT_DECL)
{
unsigned alloc
= vec_prefix::calculate_allocation (v ? &v->m_vecpfx : 0, reserve, exact);
if (!alloc)
{
::ggc_free (v);
v = NULL;
return;
}
size_t size = vec<T, A, vl_embed>::embedded_size (alloc);
size = ::ggc_round_alloc_size (size);
size_t vec_offset = sizeof (vec_prefix);
size_t elt_size = sizeof (T);
alloc = (size - vec_offset) / elt_size;
size = vec_offset + alloc * elt_size;
unsigned nelem = v ? v->length () : 0;
v = static_cast <vec<T, A, vl_embed> *> (::ggc_realloc (v, size
PASS_MEM_STAT));
v->embedded_init (alloc, nelem);
}
struct va_gc_atomic : va_gc
{
};
template<typename T,
typename A = va_heap,
typename L = typename A::default_layout>
struct GTY((user)) vec
{
};
template<typename T>
void
debug_helper (vec<T> &ref)
{
unsigned i;
for (i = 0; i < ref.length (); ++i)
{
fprintf (stderr, "[%d] = ", i);
debug_slim (ref[i]);
fputc ('\n', stderr);
}
}
template<typename T>
void
debug_helper (vec<T, va_gc> &ref)
{
unsigned i;
for (i = 0; i < ref.length (); ++i)
{
fprintf (stderr, "[%d] = ", i);
debug_slim (ref[i]);
fputc ('\n', stderr);
}
}
#define DEFINE_DEBUG_VEC(T) \
template void debug_helper (vec<T> &);		\
template void debug_helper (vec<T, va_gc> &);		\
\
DEBUG_FUNCTION void					\
debug (vec<T> &ref)					\
{							\
debug_helper <T> (ref);				\
}							\
DEBUG_FUNCTION void					\
debug (vec<T> *ptr)					\
{							\
if (ptr)						\
debug (*ptr);					\
else						\
fprintf (stderr, "<nil>\n");			\
}							\
\
DEBUG_FUNCTION void					\
debug (vec<T, va_gc> &ref)				\
{							\
debug_helper <T> (ref);				\
}							\
DEBUG_FUNCTION void					\
debug (vec<T, va_gc> *ptr)				\
{							\
if (ptr)						\
debug (*ptr);					\
else						\
fprintf (stderr, "<nil>\n");			\
}
template <typename T>
inline void
vec_default_construct (T *dst, unsigned n)
{
#ifdef BROKEN_VALUE_INITIALIZATION
memset (dst, '\0', sizeof (T) * n);
#endif
for ( ; n; ++dst, --n)
::new (static_cast<void*>(dst)) T ();
}
template <typename T>
inline void
vec_copy_construct (T *dst, const T *src, unsigned n)
{
for ( ; n; ++dst, ++src, --n)
::new (static_cast<void*>(dst)) T (*src);
}
struct vnull
{
template <typename T, typename A, typename L>
CONSTEXPR operator vec<T, A, L> () { return vec<T, A, L>(); }
};
extern vnull vNULL;
template<typename T, typename A>
struct GTY((user)) vec<T, A, vl_embed>
{
public:
unsigned allocated (void) const { return m_vecpfx.m_alloc; }
unsigned length (void) const { return m_vecpfx.m_num; }
bool is_empty (void) const { return m_vecpfx.m_num == 0; }
T *address (void) { return m_vecdata; }
const T *address (void) const { return m_vecdata; }
T *begin () { return address (); }
const T *begin () const { return address (); }
T *end () { return address () + length (); }
const T *end () const { return address () + length (); }
const T &operator[] (unsigned) const;
T &operator[] (unsigned);
T &last (void);
bool space (unsigned) const;
bool iterate (unsigned, T *) const;
bool iterate (unsigned, T **) const;
vec *copy (ALONE_CXX_MEM_STAT_INFO) const;
void splice (const vec &);
void splice (const vec *src);
T *quick_push (const T &);
T &pop (void);
void truncate (unsigned);
void quick_insert (unsigned, const T &);
void ordered_remove (unsigned);
void unordered_remove (unsigned);
void block_remove (unsigned, unsigned);
void qsort (int (*) (const void *, const void *));
T *bsearch (const void *key, int (*compar)(const void *, const void *));
unsigned lower_bound (T, bool (*)(const T &, const T &)) const;
bool contains (const T &search) const;
static size_t embedded_size (unsigned);
void embedded_init (unsigned, unsigned = 0, unsigned = 0);
void quick_grow (unsigned len);
void quick_grow_cleared (unsigned len);
template <typename, typename, typename> friend struct vec;
friend struct va_gc;
friend struct va_gc_atomic;
friend struct va_heap;
vec_prefix m_vecpfx;
T m_vecdata[1];
};
template<typename T, typename A>
inline bool
vec_safe_space (const vec<T, A, vl_embed> *v, unsigned nelems)
{
return v ? v->space (nelems) : nelems == 0;
}
template<typename T, typename A>
inline unsigned
vec_safe_length (const vec<T, A, vl_embed> *v)
{
return v ? v->length () : 0;
}
template<typename T, typename A>
inline T *
vec_safe_address (vec<T, A, vl_embed> *v)
{
return v ? v->address () : NULL;
}
template<typename T, typename A>
inline bool
vec_safe_is_empty (vec<T, A, vl_embed> *v)
{
return v ? v->is_empty () : true;
}
template<typename T, typename A>
inline bool
vec_safe_reserve (vec<T, A, vl_embed> *&v, unsigned nelems, bool exact = false
CXX_MEM_STAT_INFO)
{
bool extend = nelems ? !vec_safe_space (v, nelems) : false;
if (extend)
A::reserve (v, nelems, exact PASS_MEM_STAT);
return extend;
}
template<typename T, typename A>
inline bool
vec_safe_reserve_exact (vec<T, A, vl_embed> *&v, unsigned nelems
CXX_MEM_STAT_INFO)
{
return vec_safe_reserve (v, nelems, true PASS_MEM_STAT);
}
template<typename T, typename A>
inline void
vec_alloc (vec<T, A, vl_embed> *&v, unsigned nelems CXX_MEM_STAT_INFO)
{
v = NULL;
vec_safe_reserve (v, nelems, false PASS_MEM_STAT);
}
template<typename T, typename A>
inline void
vec_free (vec<T, A, vl_embed> *&v)
{
A::release (v);
}
template<typename T, typename A>
inline void
vec_safe_grow (vec<T, A, vl_embed> *&v, unsigned len CXX_MEM_STAT_INFO)
{
unsigned oldlen = vec_safe_length (v);
gcc_checking_assert (len >= oldlen);
vec_safe_reserve_exact (v, len - oldlen PASS_MEM_STAT);
v->quick_grow (len);
}
template<typename T, typename A>
inline void
vec_safe_grow_cleared (vec<T, A, vl_embed> *&v, unsigned len CXX_MEM_STAT_INFO)
{
unsigned oldlen = vec_safe_length (v);
vec_safe_grow (v, len PASS_MEM_STAT);
vec_default_construct (v->address () + oldlen, len - oldlen);
}
template<typename T, typename A>
inline bool
vec_safe_iterate (const vec<T, A, vl_embed> *v, unsigned ix, T **ptr)
{
if (v)
return v->iterate (ix, ptr);
else
{
*ptr = 0;
return false;
}
}
template<typename T, typename A>
inline bool
vec_safe_iterate (const vec<T, A, vl_embed> *v, unsigned ix, T *ptr)
{
if (v)
return v->iterate (ix, ptr);
else
{
*ptr = 0;
return false;
}
}
template<typename T, typename A>
inline T *
vec_safe_push (vec<T, A, vl_embed> *&v, const T &obj CXX_MEM_STAT_INFO)
{
vec_safe_reserve (v, 1, false PASS_MEM_STAT);
return v->quick_push (obj);
}
template<typename T, typename A>
inline void
vec_safe_insert (vec<T, A, vl_embed> *&v, unsigned ix, const T &obj
CXX_MEM_STAT_INFO)
{
vec_safe_reserve (v, 1, false PASS_MEM_STAT);
v->quick_insert (ix, obj);
}
template<typename T, typename A>
inline void
vec_safe_truncate (vec<T, A, vl_embed> *v, unsigned size)
{
if (v)
v->truncate (size);
}
template<typename T, typename A>
inline vec<T, A, vl_embed> *
vec_safe_copy (vec<T, A, vl_embed> *src CXX_MEM_STAT_INFO)
{
return src ? src->copy (ALONE_PASS_MEM_STAT) : NULL;
}
template<typename T, typename A>
inline void
vec_safe_splice (vec<T, A, vl_embed> *&dst, const vec<T, A, vl_embed> *src
CXX_MEM_STAT_INFO)
{
unsigned src_len = vec_safe_length (src);
if (src_len)
{
vec_safe_reserve_exact (dst, vec_safe_length (dst) + src_len
PASS_MEM_STAT);
dst->splice (*src);
}
}
template<typename T, typename A>
inline bool
vec_safe_contains (vec<T, A, vl_embed> *v, const T &search)
{
return v ? v->contains (search) : false;
}
template<typename T, typename A>
inline const T &
vec<T, A, vl_embed>::operator[] (unsigned ix) const
{
gcc_checking_assert (ix < m_vecpfx.m_num);
return m_vecdata[ix];
}
template<typename T, typename A>
inline T &
vec<T, A, vl_embed>::operator[] (unsigned ix)
{
gcc_checking_assert (ix < m_vecpfx.m_num);
return m_vecdata[ix];
}
template<typename T, typename A>
inline T &
vec<T, A, vl_embed>::last (void)
{
gcc_checking_assert (m_vecpfx.m_num > 0);
return (*this)[m_vecpfx.m_num - 1];
}
template<typename T, typename A>
inline bool
vec<T, A, vl_embed>::space (unsigned nelems) const
{
return m_vecpfx.m_alloc - m_vecpfx.m_num >= nelems;
}
template<typename T, typename A>
inline bool
vec<T, A, vl_embed>::iterate (unsigned ix, T *ptr) const
{
if (ix < m_vecpfx.m_num)
{
*ptr = m_vecdata[ix];
return true;
}
else
{
*ptr = 0;
return false;
}
}
template<typename T, typename A>
inline bool
vec<T, A, vl_embed>::iterate (unsigned ix, T **ptr) const
{
if (ix < m_vecpfx.m_num)
{
*ptr = CONST_CAST (T *, &m_vecdata[ix]);
return true;
}
else
{
*ptr = 0;
return false;
}
}
template<typename T, typename A>
inline vec<T, A, vl_embed> *
vec<T, A, vl_embed>::copy (ALONE_MEM_STAT_DECL) const
{
vec<T, A, vl_embed> *new_vec = NULL;
unsigned len = length ();
if (len)
{
vec_alloc (new_vec, len PASS_MEM_STAT);
new_vec->embedded_init (len, len);
vec_copy_construct (new_vec->address (), m_vecdata, len);
}
return new_vec;
}
template<typename T, typename A>
inline void
vec<T, A, vl_embed>::splice (const vec<T, A, vl_embed> &src)
{
unsigned len = src.length ();
if (len)
{
gcc_checking_assert (space (len));
vec_copy_construct (end (), src.address (), len);
m_vecpfx.m_num += len;
}
}
template<typename T, typename A>
inline void
vec<T, A, vl_embed>::splice (const vec<T, A, vl_embed> *src)
{
if (src)
splice (*src);
}
template<typename T, typename A>
inline T *
vec<T, A, vl_embed>::quick_push (const T &obj)
{
gcc_checking_assert (space (1));
T *slot = &m_vecdata[m_vecpfx.m_num++];
*slot = obj;
return slot;
}
template<typename T, typename A>
inline T &
vec<T, A, vl_embed>::pop (void)
{
gcc_checking_assert (length () > 0);
return m_vecdata[--m_vecpfx.m_num];
}
template<typename T, typename A>
inline void
vec<T, A, vl_embed>::truncate (unsigned size)
{
gcc_checking_assert (length () >= size);
m_vecpfx.m_num = size;
}
template<typename T, typename A>
inline void
vec<T, A, vl_embed>::quick_insert (unsigned ix, const T &obj)
{
gcc_checking_assert (length () < allocated ());
gcc_checking_assert (ix <= length ());
T *slot = &m_vecdata[ix];
memmove (slot + 1, slot, (m_vecpfx.m_num++ - ix) * sizeof (T));
*slot = obj;
}
template<typename T, typename A>
inline void
vec<T, A, vl_embed>::ordered_remove (unsigned ix)
{
gcc_checking_assert (ix < length ());
T *slot = &m_vecdata[ix];
memmove (slot, slot + 1, (--m_vecpfx.m_num - ix) * sizeof (T));
}
template<typename T, typename A>
inline void
vec<T, A, vl_embed>::unordered_remove (unsigned ix)
{
gcc_checking_assert (ix < length ());
m_vecdata[ix] = m_vecdata[--m_vecpfx.m_num];
}
template<typename T, typename A>
inline void
vec<T, A, vl_embed>::block_remove (unsigned ix, unsigned len)
{
gcc_checking_assert (ix + len <= length ());
T *slot = &m_vecdata[ix];
m_vecpfx.m_num -= len;
memmove (slot, slot + len, (m_vecpfx.m_num - ix) * sizeof (T));
}
template<typename T, typename A>
inline void
vec<T, A, vl_embed>::qsort (int (*cmp) (const void *, const void *))
{
if (length () > 1)
::qsort (address (), length (), sizeof (T), cmp);
}
template<typename T, typename A>
inline T *
vec<T, A, vl_embed>::bsearch (const void *key,
int (*compar) (const void *, const void *))
{
const void *base = this->address ();
size_t nmemb = this->length ();
size_t size = sizeof (T);
size_t l, u, idx;
const void *p;
int comparison;
l = 0;
u = nmemb;
while (l < u)
{
idx = (l + u) / 2;
p = (const void *) (((const char *) base) + (idx * size));
comparison = (*compar) (key, p);
if (comparison < 0)
u = idx;
else if (comparison > 0)
l = idx + 1;
else
return (T *)const_cast<void *>(p);
}
return NULL;
}
template<typename T, typename A>
inline bool
vec<T, A, vl_embed>::contains (const T &search) const
{
unsigned int len = length ();
for (unsigned int i = 0; i < len; i++)
if ((*this)[i] == search)
return true;
return false;
}
template<typename T, typename A>
unsigned
vec<T, A, vl_embed>::lower_bound (T obj, bool (*lessthan)(const T &, const T &))
const
{
unsigned int len = length ();
unsigned int half, middle;
unsigned int first = 0;
while (len > 0)
{
half = len / 2;
middle = first;
middle += half;
T middle_elem = (*this)[middle];
if (lessthan (middle_elem, obj))
{
first = middle;
++first;
len = len - half - 1;
}
else
len = half;
}
return first;
}
template<typename T, typename A>
inline size_t
vec<T, A, vl_embed>::embedded_size (unsigned alloc)
{
typedef vec<T, A, vl_embed> vec_embedded;
return offsetof (vec_embedded, m_vecdata) + alloc * sizeof (T);
}
template<typename T, typename A>
inline void
vec<T, A, vl_embed>::embedded_init (unsigned alloc, unsigned num, unsigned aut)
{
m_vecpfx.m_alloc = alloc;
m_vecpfx.m_using_auto_storage = aut;
m_vecpfx.m_num = num;
}
template<typename T, typename A>
inline void
vec<T, A, vl_embed>::quick_grow (unsigned len)
{
gcc_checking_assert (length () <= len && len <= m_vecpfx.m_alloc);
m_vecpfx.m_num = len;
}
template<typename T, typename A>
inline void
vec<T, A, vl_embed>::quick_grow_cleared (unsigned len)
{
unsigned oldlen = length ();
size_t growby = len - oldlen;
quick_grow (len);
if (growby != 0)
vec_default_construct (address () + oldlen, growby);
}
template<typename T>
void
gt_ggc_mx (vec<T, va_gc> *v)
{
extern void gt_ggc_mx (T &);
for (unsigned i = 0; i < v->length (); i++)
gt_ggc_mx ((*v)[i]);
}
template<typename T>
void
gt_ggc_mx (vec<T, va_gc_atomic, vl_embed> *v ATTRIBUTE_UNUSED)
{
}
template<typename T, typename A>
void
gt_pch_nx (vec<T, A, vl_embed> *v)
{
extern void gt_pch_nx (T &);
for (unsigned i = 0; i < v->length (); i++)
gt_pch_nx ((*v)[i]);
}
template<typename T, typename A>
void
gt_pch_nx (vec<T *, A, vl_embed> *v, gt_pointer_operator op, void *cookie)
{
for (unsigned i = 0; i < v->length (); i++)
op (&((*v)[i]), cookie);
}
template<typename T, typename A>
void
gt_pch_nx (vec<T, A, vl_embed> *v, gt_pointer_operator op, void *cookie)
{
extern void gt_pch_nx (T *, gt_pointer_operator, void *);
for (unsigned i = 0; i < v->length (); i++)
gt_pch_nx (&((*v)[i]), op, cookie);
}
template<typename T>
struct vec<T, va_heap, vl_ptr>
{
public:
void create (unsigned nelems CXX_MEM_STAT_INFO);
void release (void);
bool exists (void) const
{ return m_vec != NULL; }
bool is_empty (void) const
{ return m_vec ? m_vec->is_empty () : true; }
unsigned length (void) const
{ return m_vec ? m_vec->length () : 0; }
T *address (void)
{ return m_vec ? m_vec->m_vecdata : NULL; }
const T *address (void) const
{ return m_vec ? m_vec->m_vecdata : NULL; }
T *begin () { return address (); }
const T *begin () const { return address (); }
T *end () { return begin () + length (); }
const T *end () const { return begin () + length (); }
const T &operator[] (unsigned ix) const
{ return (*m_vec)[ix]; }
bool operator!=(const vec &other) const
{ return !(*this == other); }
bool operator==(const vec &other) const
{ return address () == other.address (); }
T &operator[] (unsigned ix)
{ return (*m_vec)[ix]; }
T &last (void)
{ return m_vec->last (); }
bool space (int nelems) const
{ return m_vec ? m_vec->space (nelems) : nelems == 0; }
bool iterate (unsigned ix, T *p) const;
bool iterate (unsigned ix, T **p) const;
vec copy (ALONE_CXX_MEM_STAT_INFO) const;
bool reserve (unsigned, bool = false CXX_MEM_STAT_INFO);
bool reserve_exact (unsigned CXX_MEM_STAT_INFO);
void splice (const vec &);
void safe_splice (const vec & CXX_MEM_STAT_INFO);
T *quick_push (const T &);
T *safe_push (const T &CXX_MEM_STAT_INFO);
T &pop (void);
void truncate (unsigned);
void safe_grow (unsigned CXX_MEM_STAT_INFO);
void safe_grow_cleared (unsigned CXX_MEM_STAT_INFO);
void quick_grow (unsigned);
void quick_grow_cleared (unsigned);
void quick_insert (unsigned, const T &);
void safe_insert (unsigned, const T & CXX_MEM_STAT_INFO);
void ordered_remove (unsigned);
void unordered_remove (unsigned);
void block_remove (unsigned, unsigned);
void qsort (int (*) (const void *, const void *));
T *bsearch (const void *key, int (*compar)(const void *, const void *));
unsigned lower_bound (T, bool (*)(const T &, const T &)) const;
bool contains (const T &search) const;
bool using_auto_storage () const;
vec<T, va_heap, vl_embed> *m_vec;
};
template<typename T, size_t N = 0>
class auto_vec : public vec<T, va_heap>
{
public:
auto_vec ()
{
m_auto.embedded_init (MAX (N, 2), 0, 1);
this->m_vec = &m_auto;
}
auto_vec (size_t s)
{
if (s > N)
{
this->create (s);
return;
}
m_auto.embedded_init (MAX (N, 2), 0, 1);
this->m_vec = &m_auto;
}
~auto_vec ()
{
this->release ();
}
private:
vec<T, va_heap, vl_embed> m_auto;
T m_data[MAX (N - 1, 1)];
};
template<typename T>
class auto_vec<T, 0> : public vec<T, va_heap>
{
public:
auto_vec () { this->m_vec = NULL; }
auto_vec (size_t n) { this->create (n); }
~auto_vec () { this->release (); }
};
template<typename T>
inline void
vec_alloc (vec<T> *&v, unsigned nelems CXX_MEM_STAT_INFO)
{
v = new vec<T>;
v->create (nelems PASS_MEM_STAT);
}
template<typename T>
inline void
vec_check_alloc (vec<T, va_heap> *&vec, unsigned nelems CXX_MEM_STAT_INFO)
{
if (!vec)
vec_alloc (vec, nelems PASS_MEM_STAT);
}
template<typename T>
inline void
vec_free (vec<T> *&v)
{
if (v == NULL)
return;
v->release ();
delete v;
v = NULL;
}
template<typename T>
inline bool
vec<T, va_heap, vl_ptr>::iterate (unsigned ix, T *ptr) const
{
if (m_vec)
return m_vec->iterate (ix, ptr);
else
{
*ptr = 0;
return false;
}
}
template<typename T>
inline bool
vec<T, va_heap, vl_ptr>::iterate (unsigned ix, T **ptr) const
{
if (m_vec)
return m_vec->iterate (ix, ptr);
else
{
*ptr = 0;
return false;
}
}
#define FOR_EACH_VEC_ELT(V, I, P)			\
for (I = 0; (V).iterate ((I), &(P)); ++(I))
#define FOR_EACH_VEC_SAFE_ELT(V, I, P)			\
for (I = 0; vec_safe_iterate ((V), (I), &(P)); ++(I))
#define FOR_EACH_VEC_ELT_FROM(V, I, P, FROM)		\
for (I = (FROM); (V).iterate ((I), &(P)); ++(I))
#define FOR_EACH_VEC_ELT_REVERSE(V, I, P)		\
for (I = (V).length () - 1;				\
(V).iterate ((I), &(P));				\
(I)--)
#define FOR_EACH_VEC_SAFE_ELT_REVERSE(V, I, P)		\
for (I = vec_safe_length (V) - 1;			\
vec_safe_iterate ((V), (I), &(P));	\
(I)--)
template<typename T>
inline vec<T, va_heap, vl_ptr>
vec<T, va_heap, vl_ptr>::copy (ALONE_MEM_STAT_DECL) const
{
vec<T, va_heap, vl_ptr> new_vec = vNULL;
if (length ())
new_vec.m_vec = m_vec->copy ();
return new_vec;
}
template<typename T>
inline bool
vec<T, va_heap, vl_ptr>::reserve (unsigned nelems, bool exact MEM_STAT_DECL)
{
if (space (nelems))
return false;
vec<T, va_heap, vl_embed> *oldvec = m_vec;
unsigned int oldsize = 0;
bool handle_auto_vec = m_vec && using_auto_storage ();
if (handle_auto_vec)
{
m_vec = NULL;
oldsize = oldvec->length ();
nelems += oldsize;
}
va_heap::reserve (m_vec, nelems, exact PASS_MEM_STAT);
if (handle_auto_vec)
{
vec_copy_construct (m_vec->address (), oldvec->address (), oldsize);
m_vec->m_vecpfx.m_num = oldsize;
}
return true;
}
template<typename T>
inline bool
vec<T, va_heap, vl_ptr>::reserve_exact (unsigned nelems MEM_STAT_DECL)
{
return reserve (nelems, true PASS_MEM_STAT);
}
template<typename T>
inline void
vec<T, va_heap, vl_ptr>::create (unsigned nelems MEM_STAT_DECL)
{
m_vec = NULL;
if (nelems > 0)
reserve_exact (nelems PASS_MEM_STAT);
}
template<typename T>
inline void
vec<T, va_heap, vl_ptr>::release (void)
{
if (!m_vec)
return;
if (using_auto_storage ())
{
m_vec->m_vecpfx.m_num = 0;
return;
}
va_heap::release (m_vec);
}
template<typename T>
inline void
vec<T, va_heap, vl_ptr>::splice (const vec<T, va_heap, vl_ptr> &src)
{
if (src.m_vec)
m_vec->splice (*(src.m_vec));
}
template<typename T>
inline void
vec<T, va_heap, vl_ptr>::safe_splice (const vec<T, va_heap, vl_ptr> &src
MEM_STAT_DECL)
{
if (src.length ())
{
reserve_exact (src.length ());
splice (src);
}
}
template<typename T>
inline T *
vec<T, va_heap, vl_ptr>::quick_push (const T &obj)
{
return m_vec->quick_push (obj);
}
template<typename T>
inline T *
vec<T, va_heap, vl_ptr>::safe_push (const T &obj MEM_STAT_DECL)
{
reserve (1, false PASS_MEM_STAT);
return quick_push (obj);
}
template<typename T>
inline T &
vec<T, va_heap, vl_ptr>::pop (void)
{
return m_vec->pop ();
}
template<typename T>
inline void
vec<T, va_heap, vl_ptr>::truncate (unsigned size)
{
if (m_vec)
m_vec->truncate (size);
else
gcc_checking_assert (size == 0);
}
template<typename T>
inline void
vec<T, va_heap, vl_ptr>::safe_grow (unsigned len MEM_STAT_DECL)
{
unsigned oldlen = length ();
gcc_checking_assert (oldlen <= len);
reserve_exact (len - oldlen PASS_MEM_STAT);
if (m_vec)
m_vec->quick_grow (len);
else
gcc_checking_assert (len == 0);
}
template<typename T>
inline void
vec<T, va_heap, vl_ptr>::safe_grow_cleared (unsigned len MEM_STAT_DECL)
{
unsigned oldlen = length ();
size_t growby = len - oldlen;
safe_grow (len PASS_MEM_STAT);
if (growby != 0)
vec_default_construct (address () + oldlen, growby);
}
template<typename T>
inline void
vec<T, va_heap, vl_ptr>::quick_grow (unsigned len)
{
gcc_checking_assert (m_vec);
m_vec->quick_grow (len);
}
template<typename T>
inline void
vec<T, va_heap, vl_ptr>::quick_grow_cleared (unsigned len)
{
gcc_checking_assert (m_vec);
m_vec->quick_grow_cleared (len);
}
template<typename T>
inline void
vec<T, va_heap, vl_ptr>::quick_insert (unsigned ix, const T &obj)
{
m_vec->quick_insert (ix, obj);
}
template<typename T>
inline void
vec<T, va_heap, vl_ptr>::safe_insert (unsigned ix, const T &obj MEM_STAT_DECL)
{
reserve (1, false PASS_MEM_STAT);
quick_insert (ix, obj);
}
template<typename T>
inline void
vec<T, va_heap, vl_ptr>::ordered_remove (unsigned ix)
{
m_vec->ordered_remove (ix);
}
template<typename T>
inline void
vec<T, va_heap, vl_ptr>::unordered_remove (unsigned ix)
{
m_vec->unordered_remove (ix);
}
template<typename T>
inline void
vec<T, va_heap, vl_ptr>::block_remove (unsigned ix, unsigned len)
{
m_vec->block_remove (ix, len);
}
template<typename T>
inline void
vec<T, va_heap, vl_ptr>::qsort (int (*cmp) (const void *, const void *))
{
if (m_vec)
m_vec->qsort (cmp);
}
template<typename T>
inline T *
vec<T, va_heap, vl_ptr>::bsearch (const void *key,
int (*cmp) (const void *, const void *))
{
if (m_vec)
return m_vec->bsearch (key, cmp);
return NULL;
}
template<typename T>
inline unsigned
vec<T, va_heap, vl_ptr>::lower_bound (T obj,
bool (*lessthan)(const T &, const T &))
const
{
return m_vec ? m_vec->lower_bound (obj, lessthan) : 0;
}
template<typename T>
inline bool
vec<T, va_heap, vl_ptr>::contains (const T &search) const
{
return m_vec ? m_vec->contains (search) : false;
}
template<typename T>
inline bool
vec<T, va_heap, vl_ptr>::using_auto_storage () const
{
return m_vec->m_vecpfx.m_using_auto_storage;
}
template<typename T>
inline void
release_vec_vec (vec<vec<T> > &vec)
{
for (unsigned i = 0; i < vec.length (); i++)
vec[i].release ();
vec.release ();
}
#if (GCC_VERSION >= 3000)
#pragma GCC poison m_vec m_vecpfx m_vecdata
#endif
#endif 
