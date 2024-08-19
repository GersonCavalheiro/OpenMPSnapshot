
#pragma once



#include "mappers/mapper.h"
#include "custom_utilities/interface_vector_container.h"
#include "custom_utilities/mapper_local_system.h"

#include "custom_utilities/mapping_intersection_utilities.h"
#include "custom_modelers/mapping_geometries_modeler.h"
#include "modeler/modeler_factory.h"

#include "linear_solvers/linear_solver.h"

namespace Kratos
{

class CouplingGeometryLocalSystem : public MapperLocalSystem
{
public:

explicit CouplingGeometryLocalSystem(GeometryPointerType pGeom,
const bool IsProjection,
const bool IsDualMortar,
const bool IsDestinationIsSlave
)
: mpGeom(pGeom),
mIsProjection(IsProjection),
mIsDualMortar(IsDualMortar),
mIsDestinationIsSlave(IsDestinationIsSlave)
{}

void CalculateAll(MatrixType& rLocalMappingMatrix,
EquationIdVectorType& rOriginIds,
EquationIdVectorType& rDestinationIds,
MapperLocalSystem::PairingStatus& rPairingStatus) const override;

CoordinatesArrayType& Coordinates() const override
{
KRATOS_DEBUG_ERROR_IF_NOT(mpGeom) << "Members are not intitialized!" << std::endl;
KRATOS_ERROR << "not implemented, needs checking" << std::endl;
}

MapperLocalSystemUniquePointer Create(GeometryPointerType pGeometry) const override
{
return Kratos::make_unique<CouplingGeometryLocalSystem>(pGeometry, mIsProjection, mIsDualMortar, mIsDestinationIsSlave);
}

void PairingInfo(std::ostream& rOStream, const int EchoLevel) const override {KRATOS_ERROR << "Not implemented!"<<std::endl;}

private:
GeometryPointerType mpGeom;
bool mIsProjection; 
bool mIsDualMortar = false;
bool mIsDestinationIsSlave = true;

};


template<class TSparseSpace, class TDenseSpace>
class CouplingGeometryMapper : public Mapper<TSparseSpace, TDenseSpace>
{
public:


KRATOS_CLASS_POINTER_DEFINITION(CouplingGeometryMapper);

typedef Mapper<TSparseSpace, TDenseSpace> BaseType;

typedef Kratos::unique_ptr<MapperLocalSystem> MapperLocalSystemPointer;
typedef std::vector<MapperLocalSystemPointer> MapperLocalSystemPointerVector;

typedef InterfaceVectorContainer<TSparseSpace, TDenseSpace> InterfaceVectorContainerType;
typedef Kratos::unique_ptr<InterfaceVectorContainerType> InterfaceVectorContainerPointerType;

typedef std::size_t IndexType;

typedef typename BaseType::MapperUniquePointerType MapperUniquePointerType;
typedef typename BaseType::TMappingMatrixType MappingMatrixType;
typedef Kratos::unique_ptr<MappingMatrixType> MappingMatrixUniquePointerType;

typedef LinearSolver<TSparseSpace, TDenseSpace> LinearSolverType;
typedef Kratos::shared_ptr<LinearSolverType> LinearSolverSharedPointerType;

typedef typename TSparseSpace::VectorType TSystemVectorType;
typedef Kratos::unique_ptr<TSystemVectorType> TSystemVectorUniquePointerType;


CouplingGeometryMapper(ModelPart& rModelPartOrigin,
ModelPart& rModelPartDestination)
: mrModelPartOrigin(rModelPartOrigin),
mrModelPartDestination(rModelPartDestination){}


CouplingGeometryMapper(ModelPart& rModelPartOrigin,
ModelPart& rModelPartDestination,
Parameters JsonParameters);

~CouplingGeometryMapper() override = default;


void UpdateInterface(
Kratos::Flags MappingOptions,
double SearchRadius) override
{
mpModeler->PrepareGeometryModel();

AssignInterfaceEquationIds();

KRATOS_ERROR << "Not implemented!" << std::endl;
}

void Map(
const Variable<double>& rOriginVariable,
const Variable<double>& rDestinationVariable,
Kratos::Flags MappingOptions) override
{
if (MappingOptions.Is(MapperFlags::USE_TRANSPOSE)) {
MappingOptions.Reset(MapperFlags::USE_TRANSPOSE);
MappingOptions.Set(MapperFlags::INTERNAL_USE_TRANSPOSE, true);
GetInverseMapper()->Map(rDestinationVariable, rOriginVariable, MappingOptions);
}
else if (MappingOptions.Is(MapperFlags::INTERNAL_USE_TRANSPOSE)) {
MapInternalTranspose(rOriginVariable, rDestinationVariable, MappingOptions);
}
else {
MapInternal(rOriginVariable, rDestinationVariable, MappingOptions);
}
}

void Map(
const Variable< array_1d<double, 3> >& rOriginVariable,
const Variable< array_1d<double, 3> >& rDestinationVariable,
Kratos::Flags MappingOptions) override
{
if (MappingOptions.Is(MapperFlags::USE_TRANSPOSE)) {
MappingOptions.Reset(MapperFlags::USE_TRANSPOSE);
MappingOptions.Set(MapperFlags::INTERNAL_USE_TRANSPOSE, true);
GetInverseMapper()->Map(rDestinationVariable, rOriginVariable, MappingOptions);
}
else if (MappingOptions.Is(MapperFlags::INTERNAL_USE_TRANSPOSE)) {
MapInternalTranspose(rOriginVariable, rDestinationVariable, MappingOptions);
}
else {
MapInternal(rOriginVariable, rDestinationVariable, MappingOptions);
}
}

void InverseMap(
const Variable<double>& rOriginVariable,
const Variable<double>& rDestinationVariable,
Kratos::Flags MappingOptions) override
{
if (MappingOptions.Is(MapperFlags::USE_TRANSPOSE)) {
MapInternalTranspose(rOriginVariable, rDestinationVariable, MappingOptions);
}
else {
GetInverseMapper()->Map(rDestinationVariable, rOriginVariable, MappingOptions);
}
}

void InverseMap(
const Variable< array_1d<double, 3> >& rOriginVariable,
const Variable< array_1d<double, 3> >& rDestinationVariable,
Kratos::Flags MappingOptions) override
{
if (MappingOptions.Is(MapperFlags::USE_TRANSPOSE)) {
MapInternalTranspose(rOriginVariable, rDestinationVariable, MappingOptions);
}
else {
GetInverseMapper()->Map(rDestinationVariable, rOriginVariable, MappingOptions);
}
}


MappingMatrixType& GetMappingMatrix() override
{
if (mMapperSettings["precompute_mapping_matrix"].GetBool() || mMapperSettings["dual_mortar"].GetBool()) return *(mpMappingMatrix.get());
else KRATOS_ERROR << "'precompute_mapping_matrix' or 'dual_mortar' must be 'true' in your parameters to retrieve the computed mapping matrix!" << std::endl;
}

MapperUniquePointerType Clone(ModelPart& rModelPartOrigin,
ModelPart& rModelPartDestination,
Parameters JsonParameters) const override
{
return Kratos::make_unique<CouplingGeometryMapper<TSparseSpace, TDenseSpace>>(
rModelPartOrigin,
rModelPartDestination,
JsonParameters);
}



std::string Info() const override
{
return "CouplingGeometryMapper";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "CouplingGeometryMapper";
}

void PrintData(std::ostream& rOStream) const override
{
BaseType::PrintData(rOStream);
}

ModelPart& GetInterfaceModelPartOrigin() override
{

return mpCouplingMP->GetSubModelPart("interface_origin");
}

ModelPart& GetInterfaceModelPartDestination() override
{
return mpCouplingMP->GetSubModelPart("interface_destination");
}

private:

typename Modeler::Pointer mpModeler = nullptr;

ModelPart& mrModelPartOrigin;
ModelPart& mrModelPartDestination;
ModelPart* mpCouplingMP = nullptr;
ModelPart* mpCouplingInterfaceMaster = nullptr;
ModelPart* mpCouplingInterfaceSlave = nullptr;

Parameters mMapperSettings;

MapperUniquePointerType mpInverseMapper = nullptr;

MappingMatrixUniquePointerType mpMappingMatrix;
MappingMatrixUniquePointerType mpMappingMatrixProjector;
MappingMatrixUniquePointerType mpMappingMatrixSlave;

TSystemVectorUniquePointerType mpTempVector;

MapperLocalSystemPointerVector mMapperLocalSystemsProjector;
MapperLocalSystemPointerVector mMapperLocalSystemsSlave;

InterfaceVectorContainerPointerType mpInterfaceVectorContainerMaster;
InterfaceVectorContainerPointerType mpInterfaceVectorContainerSlave;

LinearSolverSharedPointerType mpLinearSolver = nullptr;


void InitializeInterface(Kratos::Flags MappingOptions = Kratos::Flags());

void AssignInterfaceEquationIds()
{
MapperUtilities::AssignInterfaceEquationIds(mpCouplingInterfaceSlave->GetCommunicator());
MapperUtilities::AssignInterfaceEquationIds(mpCouplingInterfaceMaster->GetCommunicator());
}

void MapInternal(const Variable<double>& rOriginVariable,
const Variable<double>& rDestinationVariable,
Kratos::Flags MappingOptions);

void MapInternalTranspose(const Variable<double>& rOriginVariable,
const Variable<double>& rDestinationVariable,
Kratos::Flags MappingOptions);

void MapInternal(const Variable<array_1d<double, 3>>& rOriginVariable,
const Variable<array_1d<double, 3>>& rDestinationVariable,
Kratos::Flags MappingOptions);

void MapInternalTranspose(const Variable<array_1d<double, 3>>& rOriginVariable,
const Variable<array_1d<double, 3>>& rDestinationVariable,
Kratos::Flags MappingOptions);

void EnforceConsistencyWithScaling(
const MappingMatrixType& rInterfaceMatrixSlave,
MappingMatrixType& rInterfaceMatrixProjected,
const double scalingLimit = 1.1);

void CreateLinearSolver();

void CalculateMappingMatrixWithSolver(MappingMatrixType& rConsistentInterfaceMatrix, MappingMatrixType& rProjectedInterfaceMatrix);

Parameters GetMapperDefaultSettings() const
{
return Parameters(R"({
"echo_level"                    : 0,
"dual_mortar"                   : false,
"precompute_mapping_matrix"     : false,
"modeler_name"                  : "UNSPECIFIED",
"modeler_parameters"            : {},
"consistency_scaling"           : true,
"row_sum_tolerance"             : 1e-12,
"destination_is_slave"          : true,
"linear_solver_settings"        : {}
})");
}


MapperUniquePointerType& GetInverseMapper()
{
if (!mpInverseMapper) {
InitializeInverseMapper();
}
return mpInverseMapper;
}

void InitializeInverseMapper()
{
KRATOS_ERROR << "Inverse Mapping is not supported yet!" << std::endl;
mpInverseMapper = this->Clone(mrModelPartDestination,
mrModelPartOrigin,
mMapperSettings);
}


}; 

}  