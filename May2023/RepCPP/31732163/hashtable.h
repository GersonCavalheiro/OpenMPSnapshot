

#pragma once

#include "sync.h"
#include "tools.h"

namespace trinity {

template <typename type_t,
typename = std::enable_if_t<std::is_integral<type_t>::value> >
class Hashtable {

public:

Hashtable() = default;
Hashtable(const Hashtable& other) = delete;
Hashtable& operator=(Hashtable other) = delete;
Hashtable(Hashtable&& other) noexcept = delete;
Hashtable& operator=(Hashtable&& other) noexcept = delete;
Hashtable(size_t table_size, size_t bucket_size, size_t bucket_stride);
~Hashtable();

type_t generateKey(type_t i, type_t j, size_t scale = 1) const;
void push(type_t key, const std::initializer_list<type_t>& val);
type_t getValue(type_t v1, type_t v2, bool use_hash = false) const;
size_t getCapacity() const;
void reset();

private:

type_t** bucket   = nullptr;
int*     offset   = nullptr;
size_t   capacity = 1;
size_t   size     = 0;
size_t   stride   = 1;
int      nb_cores = 1;
};

} 

#include "hashtable.impl.h"