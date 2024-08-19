#pragma once

namespace dg
{

template <class Topology>
struct TopologyTraits{
typedef typename Topology::memory_category memory_category; 
typedef typename Topology::dimensionality dimensionality; 
typedef typename Topology::value_type value_type; 
};

struct MPITag{}; 
struct SharedTag{}; 

struct OneDimensionalTag{}; 
struct TwoDimensionalTag{}; 
struct ThreeDimensionalTag{}; 

}
