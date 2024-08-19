#pragma once

#include <cstdio> 
#include <cassert> 
#include <cstdint> 
#include <vector> 
#include <algorithm> 

#define debug_printf(...)

template <typename T>
class data_list 
{
private:
std::vector<T> _data;
std::vector<T*> _ptrs;
std::vector<uint32_t> _m;
size_t _mem;
uint32_t _n;
uint32_t _max_m;
public:

template <typename int_t>
data_list(uint32_t const n, int_t const ms[], T const init_value=T(0)) 
: _ptrs(n, nullptr), _m(n), _mem(0), _n(n), _max_m(0) {
assert(n == _n); 
size_t num{0}; 
for (uint32_t i = 0; i < n; ++i) {
auto const m = uint32_t(std::max(ms[i], int_t(0)));
_m[i] = m;
assert(ms[i] == _m[i]); 
_max_m = std::max(_max_m, m);
num += _m[i];
} 
_mem = num*sizeof(T);
debug_printf("# data_list() constructor tries to allocate %.6f MByte\n", _mem*1e-6);
_data = std::vector<T>(num, init_value);
num = 0;
for (uint32_t i = 0; i < n; ++i) {
_ptrs[i] = &_data[num];
num += _m[i];
} 
assert(num*sizeof(T) == _mem); 
} 

template <typename int_t>
data_list(std::vector<int_t> const & ms, T const init_value=T(0)) 
: data_list(ms.size(), ms.data(), init_value) {} 

data_list(void) : _data(0), _ptrs(0), _m(0), _mem(0), _n(0), _max_m(0) {} 

~data_list() {
debug_printf("# ~data_list() destructor tries to free %.6f MByte\n", _mem*1e-6);
} 

data_list(data_list<T> && rhs) = delete; 

data_list(data_list<T> const & rhs) = delete; 
data_list& operator= (data_list<T> const & rhs) = delete; 

data_list& operator= (data_list<T> && rhs) {
debug_printf("# data_list& operator= (data_list<T> && rhs) transfers %.6f MByte\n", rhs._mem*1e-6);
_data.swap(rhs._data);
_ptrs.swap(rhs._ptrs);
_m.swap(rhs._m);
_mem   = rhs._mem;    rhs._mem   = 0;
_n     = rhs._n;      rhs._n     = 0;
_max_m = rhs._max_m;  rhs._max_m = 0;
return *this;
} 


T const & operator () (uint32_t const i, uint32_t const j) const { return _ptrs[i][j]; } 
T       & operator () (uint32_t const i, uint32_t const j)       { return _ptrs[i][j]; } 

T const & at(uint32_t const i, uint32_t const j) const { assert(i < _n); assert(j < _m[i]); return _ptrs[i][j]; }
T       & at(uint32_t const i, uint32_t const j)       { assert(i < _n); assert(j < _m[i]); return _ptrs[i][j]; }

T* operator[] (uint32_t const i) const { assert(i < _n); return _ptrs[i]; }

T const *const * data() const { return _ptrs.data(); } 
T       *const * data()       { return _ptrs.data(); }

uint32_t nrows() const { return _n; } 
uint32_t mcols() const { return _max_m; } 
uint32_t const * m() const { return _m.data(); } 
size_t fill(T const value={0}) { std::fill(_data.begin(), _data.end(), value); return _data.size(); } 

}; 

#undef debug_printf
