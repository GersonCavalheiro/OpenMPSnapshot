
#pragma once

#include "grmhd.hpp"

using namespace parthenon;

namespace GRMHD {


inline VariablePack<Real> PackMHDPrims(MeshBlockData<Real> *rc, PackIndexMap& prims_map, bool coarse=false)
{
auto pmb = rc->GetBlockPointer();
MetadataFlag isPrimitive = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
MetadataFlag isMHD = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("MHDFlag");
return rc->PackVariables({isPrimitive, isMHD}, prims_map, coarse);
}
inline MeshBlockPack<VariablePack<Real>> PackMHDPrims(MeshData<Real> *md, PackIndexMap& prims_map, bool coarse=false)
{
auto pmb = md->GetBlockData(0)->GetBlockPointer();
MetadataFlag isPrimitive = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
MetadataFlag isMHD = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("MHDFlag");
return md->PackVariables(std::vector<MetadataFlag>{isPrimitive, isMHD}, prims_map, coarse);
}

inline VariablePack<Real> PackMHDCons(MeshBlockData<Real> *rc, PackIndexMap& cons_map, bool coarse=false)
{
auto pmb = rc->GetBlockPointer();
MetadataFlag isMHD = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("MHDFlag");
return rc->PackVariables({Metadata::Conserved, isMHD}, cons_map, coarse);
}
inline MeshBlockPack<VariablePack<Real>> PackMHDCons(MeshData<Real> *md, PackIndexMap& cons_map, bool coarse=false)
{
auto pmb = md->GetBlockData(0)->GetBlockPointer();
MetadataFlag isMHD = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("MHDFlag");
return md->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved, isMHD}, cons_map, coarse);
}

inline VariablePack<Real> PackHDPrims(MeshBlockData<Real> *rc, PackIndexMap& prims_map, bool coarse=false)
{
auto pmb = rc->GetBlockPointer();
MetadataFlag isPrimitive = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
MetadataFlag isHD = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("HDFlag");
return rc->PackVariables({isPrimitive, isHD}, prims_map, coarse);
}
inline MeshBlockPack<VariablePack<Real>> PackHDPrims(MeshData<Real> *md, PackIndexMap& prims_map, bool coarse=false)
{
auto pmb = md->GetBlockData(0)->GetBlockPointer();
MetadataFlag isPrimitive = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
MetadataFlag isHD = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("HDFlag");
return md->PackVariables(std::vector<MetadataFlag>{isPrimitive, isHD}, prims_map, coarse);
}
template<typename T>
inline VariablePack<Real> PackHDPrims(T data) { PackIndexMap nop; return PackHDPrims(data, nop); }

inline VariablePack<Real> PackHDCons(MeshBlockData<Real> *rc, PackIndexMap& cons_map, bool coarse=false)
{
auto pmb = rc->GetBlockPointer();
MetadataFlag isHD = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("HDFlag");
return rc->PackVariables({Metadata::Conserved, isHD}, cons_map, coarse);
}
inline MeshBlockPack<VariablePack<Real>> PackHDCons(MeshData<Real> *md, PackIndexMap& cons_map, bool coarse=false)
{
auto pmb = md->GetBlockData(0)->GetBlockPointer();
MetadataFlag isHD = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("HDFlag");
return md->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved, isHD}, cons_map, coarse);
}


} 
