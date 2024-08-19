#pragma once

#include <Config.h>
#include <iostream>

namespace CompactNSearch
{
class NeighborhoodSearch;


class PointSet
{

public:


PointSet(PointSet const& other)
{
*this = other;
}


PointSet& operator=(PointSet const& other)
{
m_x = other.m_x;
m_n = other.m_n;
m_dynamic = other.m_dynamic;
m_user_data = other.m_user_data;

m_neighbors = other.m_neighbors;
m_keys = other.m_keys;
m_old_keys = other.m_old_keys;

m_sort_table = other.m_sort_table;

m_locks.resize(other.m_locks.size());
for (unsigned int i = 0; i < other.m_locks.size(); ++i)
{
m_locks[i].resize(other.m_locks[i].size());
}

return *this;
}


std::size_t n_neighbors(unsigned int point_set, unsigned int i) const 
{
return static_cast<unsigned int>(m_neighbors[point_set][i].size());
}


unsigned int neighbor(unsigned int point_set, unsigned int i, unsigned int k) const 
{
return m_neighbors[point_set][i][k];
}


const std::vector<unsigned int>& neighbor_list(unsigned int point_set, unsigned int i) const
{
return m_neighbors[point_set][i];
}


std::size_t n_points() const { return m_n; }


bool is_dynamic() const { return m_dynamic; }


void set_dynamic(bool v) { m_dynamic = v; }


void *get_user_data() { return m_user_data;  }


template <typename T>
void sort_field(T* lst) const;

private:

friend NeighborhoodSearch;
PointSet(Real const* x, std::size_t n, bool dynamic, void *user_data = nullptr)
: m_x(x), m_n(n), m_dynamic(dynamic), m_user_data(user_data), m_neighbors(n)
, m_keys(n, {
std::numeric_limits<int>::lowest(),
std::numeric_limits<int>::lowest(),
std::numeric_limits<int>::lowest() })
{
m_old_keys = m_keys;
}

void resize(Real const* x, std::size_t n)
{ 
m_x = x;
m_n = n; 
m_keys.resize(n, {
std::numeric_limits<int>::lowest(),
std::numeric_limits<int>::lowest(),
std::numeric_limits<int>::lowest() }); 
m_old_keys.resize(n, {
std::numeric_limits<int>::lowest(),
std::numeric_limits<int>::lowest(),
std::numeric_limits<int>::lowest() });
m_neighbors.resize(n);
}

Real const* point(unsigned int i) const { return &m_x[3*i]; }

private:

Real const* m_x;
std::size_t m_n;
bool m_dynamic;
void *m_user_data;

std::vector<std::vector<std::vector<unsigned int>>> m_neighbors;

std::vector<HashKey> m_keys, m_old_keys;
std::vector<std::vector<Spinlock>> m_locks;
std::vector<unsigned int> m_sort_table;
};

template <typename T> void
PointSet::sort_field(T* lst) const
{
if (m_sort_table.empty())
{
std::cerr << "WARNING: No sort table was generated for the current point set. "
<< "First invoke the method 'z_sort' of the class 'NeighborhoodSearch.'" << std::endl;
return;
}

std::vector<T> tmp(lst, lst + m_sort_table.size());
std::transform(m_sort_table.begin(), m_sort_table.end(), 
#ifdef _MSC_VER
stdext::unchecked_array_iterator<T*>(lst),
#else
lst,
#endif
[&](int i){ return tmp[i]; });
}
}

