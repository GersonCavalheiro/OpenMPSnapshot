




#ifndef FF_SVECTOR_HPP
#define FF_SVECTOR_HPP



#include <stdlib.h>
#include <new>

#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && (((__GNUC__ * 100) + __GNUC_MINOR__) >= 800)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif

namespace ff {





template <typename T>
class svector {


enum {SVECTOR_CHUNK=1024};
public:
typedef T* iterator;
typedef const T* const_iterator;
typedef T vector_type;


svector(size_t chunk=SVECTOR_CHUNK):first(NULL),len(0),cap(0),chunk(chunk) {
reserve(chunk);
}


svector(const svector & v):first(NULL),len(0),cap(0),chunk(v.chunk) {
if(v.len) {
const_iterator i1=v.begin();
const_iterator i2=v.end();
first=(vector_type*)::malloc((i2-i1)*sizeof(vector_type));
while(i1!=i2) push_back(*(i1++));
return;
}
reserve(v.chunk);
}


svector(const_iterator i1,const_iterator i2):first(0),len(0),cap(0),chunk(SVECTOR_CHUNK) {
first=(vector_type*)::malloc((i2-i1)*sizeof(vector_type));
while(i1!=i2) push_back(*(i1++));
}



svector(svector &&v) {
first = v.first;
len   = v.len;
cap   = v.cap;
chunk = v.chunk;

v.first = nullptr;
}


~svector() { 
if(first) {  
clear(); 
::free(first); 
first=NULL;
}  
}


svector& operator=(const svector & v) {
if(!v.len) clear();
else {
const_iterator i1=v.begin();
const_iterator i2=v.end();
if (first) { clear(); ::free(first); }
first=(vector_type*)::malloc((i2-i1)*sizeof(vector_type));
while(i1!=i2) push_back(*(i1++));
}
return *this;
}
svector& operator=(svector && v) {
if (this != &v) {
if (first) { clear(); ::free(first); }
first=v.first;
len  =v.len;
cap  =v.cap;
chunk=v.chunk;

v.first = nullptr;
}
return *this;
}


svector& operator+=(const svector & v) {
const_iterator i1=v.begin();
const_iterator i2=v.end();
while(i1!=i2) push_back(*(i1++));
return *this;
}


inline void reserve(size_t newcapacity) {
if(newcapacity<=cap) return;
if (first==NULL)
first=(vector_type*)::malloc(sizeof(vector_type)*newcapacity);
else 
first=(vector_type*)::realloc(first,sizeof(vector_type)*newcapacity);
cap = newcapacity;
}


inline void resize(size_t newsize) {
if (len >= newsize) {
while(len>newsize) pop_back();
return;
}
reserve(newsize);
while(len<newsize)
new (first + len++) vector_type();
}


inline void push_back(const vector_type & elem) {
if (len==cap) reserve(cap+chunk);	    
new (first + len++) vector_type(elem); 
}


inline void pop_back() { (first + --len)->~vector_type();  }


inline vector_type& back() const { 
return first[len-1]; 
}


inline vector_type& front() {
return first[0];
}


inline const vector_type& front() const {
return first[0];
}


inline iterator erase(iterator where) {
iterator i1=begin();
iterator i2=end();
while(i1!=i2) {
if (i1==where) { --len; break;}
else i1++;
}

for(iterator i3=i1++; i1!=i2; ++i3, ++i1) 
*i3 = *i1;

return begin();
}

inline iterator insert(iterator where, const vector_type & elem) {
iterator i1=begin();
iterator i2=end();
if (where == i2) {
push_back(elem);
return end();
}
while(i1!=i2) {
if (i1==where) {
++len;
if (len==cap) reserve(cap+chunk);
break;
}
else i1++;
}
for(iterator i3=i2+1; i2>=i1; --i2, --i3) 
*i3 = *i2;
size_t pos=(i1-begin());
new (first + pos) vector_type(elem);
return begin()+pos;
}



inline size_t size() const { return len; }


inline bool   empty() const { return (len==0);}


inline size_t capacity() const { return cap;}


inline void clear() { while(size()>0) pop_back(); }


iterator begin() { return first; }


iterator end() { return first+len; }


const_iterator begin() const { return first; }


const_iterator end() const { return first+len; }





vector_type& operator[](size_t n) { 
reserve(n+1);
return first[n]; 
}


const vector_type& operator[](size_t n) const { return first[n]; }

private:
vector_type * first;
size_t        len;   
size_t        cap;   
size_t        chunk;
};



} 

#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && (((__GNUC__ * 100) + __GNUC_MINOR__) >= 800)
#pragma GCC diagnostic pop
#endif



#endif 
