#pragma once

#include "Config.h"
#include "DataStructures.h"
#include "PointSet.h"

#include <unordered_map>

namespace CompactNSearch
{

struct NeighborhoodSearchNotInitialized : public std::exception
{
virtual char const* what() const noexcept override { return "Neighborhood search was not initialized."; }
};


class NeighborhoodSearch
{

public:


NeighborhoodSearch(Real r, bool erase_empty_cells = false);


virtual ~NeighborhoodSearch() = default;


PointSet const& point_set(unsigned int i) const { return m_point_sets[i];     }


PointSet      & point_set(unsigned int i)       { return m_point_sets[i];     }



std::size_t  n_point_sets()               const { return m_point_sets.size(); }


std::vector<PointSet> const& point_sets() const { return m_point_sets;        }


std::vector<PointSet>      & point_sets()       { return m_point_sets;        }


void resize_point_set(unsigned int i, Real const* x, std::size_t n);


unsigned int add_point_set(Real const* x, std::size_t n, bool is_dynamic = true,
bool search_neighbors = true, bool find_neighbors = true, void *user_data = nullptr)
{ 
m_point_sets.push_back({x, n, is_dynamic, user_data});
m_activation_table.add_point_set(search_neighbors, find_neighbors);
return static_cast<unsigned int>(m_point_sets.size() - 1);
}


void find_neighbors(bool points_changed = true);


void find_neighbors(unsigned int point_set_id, unsigned int point_index, std::vector<std::vector<unsigned int>> &neighbors);


void find_neighbors(Real const* x, std::vector<std::vector<unsigned int>> &neighbors);


void update_point_sets();


void update_activation_table();



void z_sort();


Real radius() const { return std::sqrt(m_r2); }


void set_radius(Real r) 
{ 
m_r2 = r * r; 
m_inv_cell_size = static_cast<Real>(1.0 / r);
m_initialized = false;
}


void set_active(unsigned int i, unsigned int j, bool active)
{
m_activation_table.set_active(i, j, active);
m_initialized = false;
}


void set_active(unsigned int i, bool search_neighbors = true, bool find_neighbors = true)
{
m_activation_table.set_active(i, search_neighbors, find_neighbors);
m_initialized = false;
}


void set_active(bool active)
{
m_activation_table.set_active(active);
m_initialized = false;
}


bool is_active(unsigned int i, unsigned int j) const
{
return m_activation_table.is_active(i, j);
}

private:

void init();
void update_hash_table(std::vector<unsigned int>& to_delete);
void erase_empty_entries(std::vector<unsigned int> const& to_delete);
void query();
void query(unsigned int point_set_id, unsigned int point_index, std::vector<std::vector<unsigned int>> &neighbors);
void query(Real const* xa, std::vector<std::vector<unsigned int>> &neighbors);

HashKey cell_index(Real const* x) const;

private:


std::vector<PointSet> m_point_sets;
ActivationTable m_activation_table, m_old_activation_table;

Real m_inv_cell_size;
Real m_r2;
std::unordered_map<HashKey, unsigned int, SpatialHasher> m_map;
std::vector<HashEntry> m_entries;

bool m_erase_empty_cells;
bool m_initialized;
};

}
