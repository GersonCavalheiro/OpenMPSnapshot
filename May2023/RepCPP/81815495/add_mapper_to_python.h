
#pragma once



#include "includes/define_python.h"
#include "mappers/mapper.h"
#include "factories/mapper_factory.h"
#include "mappers/mapper_flags.h"

namespace Kratos::Python
{
namespace {

template<class TSparseSpace, class TDenseSpace>
inline void UpdateInterfaceWithoutArgs(Mapper<TSparseSpace, TDenseSpace>& dummy)
{
Kratos::Flags dummy_flags = Kratos::Flags();
double dummy_search_radius = -1.0f;
dummy.UpdateInterface(dummy_flags, dummy_search_radius);
}

template<class TSparseSpace, class TDenseSpace>
inline void UpdateInterfaceWithOptions(Mapper<TSparseSpace, TDenseSpace>& dummy, Kratos::Flags options)
{
double dummy_search_radius = -1.0f;
dummy.UpdateInterface(options, dummy_search_radius);
}

template<class TSparseSpace, class TDenseSpace>
inline void UpdateInterfaceWithSearchRadius(Mapper<TSparseSpace, TDenseSpace>& dummy, double search_radius)
{
Kratos::Flags dummy_flags = Kratos::Flags();
dummy.UpdateInterface(dummy_flags, search_radius);
}


template<class TSparseSpace, class TDenseSpace>
inline void MapWithoutOptionsScalar(Mapper<TSparseSpace, TDenseSpace>& dummy,
const Variable<double>& origin_variable,
const Variable<double>& destination_variable)
{
Kratos::Flags dummy_flags = Kratos::Flags();
dummy.Map(origin_variable, destination_variable, dummy_flags);
}

template<class TSparseSpace, class TDenseSpace>
inline void MapWithoutOptionsVector(Mapper<TSparseSpace, TDenseSpace>& dummy,
const Variable< array_1d<double, 3> >& origin_variable,
const Variable< array_1d<double, 3> >& destination_variable)
{
Kratos::Flags dummy_flags = Kratos::Flags();
dummy.Map(origin_variable, destination_variable, dummy_flags);
}

template<class TSparseSpace, class TDenseSpace>
inline void InverseMapWithoutOptionsScalar(Mapper<TSparseSpace, TDenseSpace>& dummy,
const Variable<double>& origin_variable,
const Variable<double>& destination_variable)
{
Kratos::Flags dummy_flags = Kratos::Flags();
dummy.InverseMap(origin_variable, destination_variable, dummy_flags);
}

template<class TSparseSpace, class TDenseSpace>
inline void InverseMapWithoutOptionsVector(Mapper<TSparseSpace, TDenseSpace>& dummy,
const Variable< array_1d<double, 3> >& origin_variable,
const Variable< array_1d<double, 3> >& destination_variable)
{
Kratos::Flags dummy_flags = Kratos::Flags();
dummy.InverseMap(origin_variable, destination_variable, dummy_flags);
}


template<class TSparseSpace, class TDenseSpace>
inline void MapWithOptionsScalar(Mapper<TSparseSpace, TDenseSpace>& dummy,
const Variable<double>& origin_variable,
const Variable<double>& destination_variable,
Kratos::Flags MappingOptions)
{
dummy.Map(origin_variable, destination_variable, MappingOptions);
}

template<class TSparseSpace, class TDenseSpace>
inline void MapWithOptionsVector(Mapper<TSparseSpace, TDenseSpace>& dummy,
const Variable< array_1d<double, 3> >& origin_variable,
const Variable< array_1d<double, 3> >& destination_variable,
Kratos::Flags MappingOptions)
{
dummy.Map(origin_variable, destination_variable, MappingOptions);
}

template<class TSparseSpace, class TDenseSpace>
inline void InverseMapWithOptionsScalar(Mapper<TSparseSpace, TDenseSpace>& dummy,
const Variable<double>& origin_variable,
const Variable<double>& destination_variable,
Kratos::Flags MappingOptions)
{
dummy.InverseMap(origin_variable, destination_variable, MappingOptions);
}

template<class TSparseSpace, class TDenseSpace>
inline void InverseMapWithOptionsVector(Mapper<TSparseSpace, TDenseSpace>& dummy,
const Variable< array_1d<double, 3> >& origin_variable,
const Variable< array_1d<double, 3> >& destination_variable,
Kratos::Flags MappingOptions)
{
dummy.InverseMap(origin_variable, destination_variable, MappingOptions);
}


template<class TSparseSpace, class TDenseSpace>
void ExposeMapperToPython(pybind11::module& m)
{
namespace py = pybind11;
const std::string mapper_name = TSparseSpace::IsDistributed() ? "MPIMapper" : "Mapper";
typedef Mapper<TSparseSpace, TDenseSpace> MapperType;
const auto mapper
= py::class_< MapperType, typename MapperType::Pointer >(m, mapper_name.c_str())
.def("UpdateInterface",     UpdateInterfaceWithoutArgs<TSparseSpace, TDenseSpace>)
.def("UpdateInterface",     UpdateInterfaceWithOptions<TSparseSpace, TDenseSpace>)
.def("UpdateInterface",     UpdateInterfaceWithSearchRadius<TSparseSpace, TDenseSpace>)
.def("UpdateInterface",     &MapperType::UpdateInterface) 

.def("Map",                 MapWithoutOptionsScalar<TSparseSpace, TDenseSpace>)
.def("Map",                 MapWithoutOptionsVector<TSparseSpace, TDenseSpace>)
.def("Map",                 MapWithOptionsScalar<TSparseSpace, TDenseSpace>)
.def("Map",                 MapWithOptionsVector<TSparseSpace, TDenseSpace>)

.def("InverseMap",          InverseMapWithoutOptionsScalar<TSparseSpace, TDenseSpace>)
.def("InverseMap",          InverseMapWithoutOptionsVector<TSparseSpace, TDenseSpace>)
.def("InverseMap",          InverseMapWithOptionsScalar<TSparseSpace, TDenseSpace>)
.def("InverseMap",          InverseMapWithOptionsVector<TSparseSpace, TDenseSpace>)

.def("GetMappingMatrix",    &MapperType::GetMappingMatrix, py::return_value_policy::reference_internal)
.def("GetInterfaceModelPartOrigin", &MapperType::GetInterfaceModelPartOrigin, py::return_value_policy::reference_internal)
.def("GetInterfaceModelPartDestination", &MapperType::GetInterfaceModelPartDestination, py::return_value_policy::reference_internal)

.def("AreMeshesConforming", &MapperType::AreMeshesConforming)

.def("__str__",             PrintObject<MapperType>)
;

mapper.attr("SWAP_SIGN")           = MapperFlags::SWAP_SIGN;
mapper.attr("ADD_VALUES")          = MapperFlags::ADD_VALUES;
mapper.attr("REMESHED")            = MapperFlags::REMESHED;
mapper.attr("USE_TRANSPOSE")       = MapperFlags::USE_TRANSPOSE;
mapper.attr("TO_NON_HISTORICAL")   = MapperFlags::TO_NON_HISTORICAL;
mapper.attr("FROM_NON_HISTORICAL") = MapperFlags::FROM_NON_HISTORICAL;
}

} 

template<class TSparseSpace, class TDenseSpace>
void AddMappingToPython(pybind11::module& m)
{
ExposeMapperToPython<TSparseSpace, TDenseSpace>(m);

const std::string mapper_factory_name = TSparseSpace::IsDistributed() ? "MPIMapperFactory" : "MapperFactory";
typedef MapperFactory<TSparseSpace, TDenseSpace> MapperFactoryType;
pybind11::class_<MapperFactoryType, typename MapperFactoryType::Pointer>(m, mapper_factory_name.c_str())
.def_static("CreateMapper",             &MapperFactoryType::CreateMapper)
.def_static("HasMapper",                &MapperFactoryType::HasMapper)
.def_static("GetRegisteredMapperNames", &MapperFactoryType::GetRegisteredMapperNames)
;
}

void AddMapperToPython(pybind11::module& m);

}  