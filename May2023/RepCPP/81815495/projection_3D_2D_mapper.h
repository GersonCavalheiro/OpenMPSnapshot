
#pragma once



#include "interpolative_mapper_base.h"
#include "custom_mappers/nearest_neighbor_mapper.h"
#include "custom_mappers/nearest_element_mapper.h"
#include "custom_mappers/barycentric_mapper.h"
#include "utilities/geometrical_projection_utilities.h"
#include "utilities/parallel_utilities.h"

namespace Kratos
{


using NodeType = Node;
using GeometryType = Geometry<NodeType>;




GeometryType::Pointer GetGeometryFromModelPart(const ModelPart& rModelPart)
{
return (rModelPart.NumberOfElements() > 0 ? rModelPart.ElementsBegin()->pGetGeometry() : rModelPart.NumberOfConditions() > 0 ? rModelPart.ConditionsBegin()->pGetGeometry() : nullptr);
}


int DeterminePartitionWithEntities(const ModelPart& rModelPart)
{
auto p_geometry = GetGeometryFromModelPart(rModelPart);
const int partition_entity = (p_geometry != nullptr) ? rModelPart.GetCommunicator().GetDataCommunicator().Rank() : -1;
return rModelPart.GetCommunicator().GetDataCommunicator().MaxAll(partition_entity);
}


unsigned int DetermineModelPartMaximumLocalDimension(ModelPart& rModelPart)
{
auto p_geometry = GetGeometryFromModelPart(rModelPart);
const unsigned int local_space_dimension = (p_geometry == nullptr) ? 0 : p_geometry->LocalSpaceDimension();
return rModelPart.GetCommunicator().GetDataCommunicator().MaxAll(local_space_dimension);
}


ModelPart& Determine2DModelPart(ModelPart& rFirstModelPart, ModelPart& rSecondModelPart)
{
const unsigned int max_local_space_dimension_1 = DetermineModelPartMaximumLocalDimension(rFirstModelPart);
const unsigned int max_local_space_dimension_2 = DetermineModelPartMaximumLocalDimension(rSecondModelPart);
KRATOS_ERROR_IF(max_local_space_dimension_1 == 3 && max_local_space_dimension_2 == 3) << "Both model parts are 3D" << std::endl;
KRATOS_ERROR_IF(max_local_space_dimension_1 == 1 || max_local_space_dimension_2 == 1) << "One model part is 1D, not compatible" << std::endl;
KRATOS_ERROR_IF(max_local_space_dimension_1 == 0 || max_local_space_dimension_2 == 0) << "Impossible to determine local space dimension in at least one model part" << std::endl;
if (max_local_space_dimension_1 == 2) {
return rFirstModelPart;
} else if (max_local_space_dimension_2 == 2) {
return rSecondModelPart;
} else { 
KRATOS_ERROR << "Impossible to detect 2D model part" << std::endl;
}
return rFirstModelPart;
}


ModelPart& Determine3DModelPart(ModelPart& rFirstModelPart, ModelPart& rSecondModelPart)
{
const unsigned int max_local_space_dimension_1 = DetermineModelPartMaximumLocalDimension(rFirstModelPart);
const unsigned int max_local_space_dimension_2 = DetermineModelPartMaximumLocalDimension(rSecondModelPart);
KRATOS_ERROR_IF(max_local_space_dimension_1 == 3 && max_local_space_dimension_2 == 3) << "Both model parts are 3D" << std::endl;
KRATOS_ERROR_IF(max_local_space_dimension_1 == 1 || max_local_space_dimension_2 == 1) << "One model part is 1D, not compatible" << std::endl;
KRATOS_ERROR_IF(max_local_space_dimension_1 == 0 || max_local_space_dimension_2 == 0) << "Impossible to determine local space dimension in at least one model part" << std::endl;
if (max_local_space_dimension_1 == 3) {
return rFirstModelPart;
} else if (max_local_space_dimension_2 == 3) {
return rSecondModelPart;
} else { 
KRATOS_ERROR << "Impossible to detect 3D model part" << std::endl;
}
return rFirstModelPart;
}



template<class TSparseSpace, class TDenseSpace, class TMapperBackend>
class Projection3D2DMapper
: public InterpolativeMapperBase<TSparseSpace, TDenseSpace, TMapperBackend>
{
public:

KRATOS_CLASS_POINTER_DEFINITION(Projection3D2DMapper);

typedef InterpolativeMapperBase<TSparseSpace, TDenseSpace, TMapperBackend> BaseType;
typedef Kratos::unique_ptr<BaseType> BaseMapperUniquePointerType;
typedef typename BaseType::TMappingMatrixType TMappingMatrixType;
typedef typename BaseType::MapperUniquePointerType MapperUniquePointerType;

typedef typename TMapperBackend::InterfaceCommunicatorType InterfaceCommunicatorType;
typedef typename InterfaceCommunicator::MapperInterfaceInfoUniquePointerType MapperInterfaceInfoUniquePointerType;

typedef NearestNeighborMapper<TSparseSpace, TDenseSpace, TMapperBackend> NearestNeighborMapperType;
typedef NearestElementMapper<TSparseSpace, TDenseSpace, TMapperBackend>   NearestElementMapperType;
typedef BarycentricMapper<TSparseSpace, TDenseSpace, TMapperBackend>         BarycentricMapperType;



enum class EntityTypeMesh
{
NONE,
CONDITIONS,
ELEMENTS
};


Projection3D2DMapper(
ModelPart& rModelPartOrigin,
ModelPart& rModelPartDestination
) : BaseType(rModelPartOrigin, rModelPartDestination),
mr2DModelPart(rModelPartOrigin),
mr3DModelPart(rModelPartDestination)
{
}

Projection3D2DMapper(
ModelPart& rModelPartOrigin,
ModelPart& rModelPartDestination,
Parameters JsonParameters
) : BaseType(rModelPartOrigin, rModelPartDestination, JsonParameters),
mr2DModelPart(Determine2DModelPart(rModelPartOrigin, rModelPartDestination)),
mr3DModelPart(Determine3DModelPart(rModelPartOrigin, rModelPartDestination))
{
KRATOS_TRY;

this->ValidateInput();

mCopiedParameters = JsonParameters.Clone();

CheckOriginIs2D();

mMetaMapperType = mCopiedParameters["base_mapper"].GetString();

if (mOriginIs2D) {
GetEntityMeshType();

GetNormalAndReferencePlane();

MoveModelParts();
}

mCopiedParameters.RemoveValue("base_mapper");

CreateBaseMapper();

if (mOriginIs2D) {
UnMoveModelParts();
}

this->Initialize();

BaseType::mpMappingMatrix = Kratos::make_unique<TMappingMatrixType>(mpBaseMapper->GetMappingMatrix());

KRATOS_CATCH("");
}

~Projection3D2DMapper() override = default;




void UpdateInterface(
Kratos::Flags MappingOptions,
double SearchRadius
) override
{
KRATOS_TRY;

if (mOriginIs2D) {
MoveModelParts();
}

CreateBaseMapper();

mpBaseMapper->UpdateInterface(MappingOptions, SearchRadius);

if (mOriginIs2D) {
UnMoveModelParts();
}

BaseType::UpdateInterface(MappingOptions, SearchRadius);

BaseType::mpMappingMatrix = Kratos::make_unique<TMappingMatrixType>(mpBaseMapper->GetMappingMatrix());

KRATOS_CATCH("");
}


MapperUniquePointerType Clone(
ModelPart& rModelPartOrigin,
ModelPart& rModelPartDestination,
Parameters JsonParameters
) const override
{
KRATOS_TRY;

return Kratos::make_unique<Projection3D2DMapper<TSparseSpace, TDenseSpace, TMapperBackend>>(rModelPartOrigin, rModelPartDestination, JsonParameters);

KRATOS_CATCH("");
}



ModelPart& Get2DModelPart()
{
return mr2DModelPart;
}


ModelPart& Get3DModelPart()
{
return mr3DModelPart;
}





std::string Info() const override
{
return "Projection3D2DMapper";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "Projection3D2DMapper";
}

void PrintData(std::ostream& rOStream) const override
{
BaseType::PrintData(rOStream);
}

private:


ModelPart& mr2DModelPart;                   
ModelPart& mr3DModelPart;                   
BaseMapperUniquePointerType mpBaseMapper;   
array_1d<double, 3> mNormalPlane;           
Point mPointPlane;                          
Parameters mCopiedParameters;               
std::string mMetaMapperType;                
EntityTypeMesh mEntityTypeMesh;             
bool mOriginIs2D;                           



void CheckOriginIs2D()
{
KRATOS_TRY;

const auto* p_origin_model_part = &(this->GetOriginModelPart());
const auto* p_2d_model_part = &(this->Get2DModelPart());
mOriginIs2D = p_origin_model_part == p_2d_model_part ? true : false;

KRATOS_CATCH("");
}


void GetEntityMeshType()
{
KRATOS_TRY;

const auto& r_2d_model_part = this->Get2DModelPart();
if (r_2d_model_part.NumberOfConditions() > 0) {
mEntityTypeMesh = EntityTypeMesh::CONDITIONS;
} else if (r_2d_model_part.NumberOfElements() > 0) {
mEntityTypeMesh = EntityTypeMesh::ELEMENTS;
} else {
mEntityTypeMesh = EntityTypeMesh::NONE; 
}

KRATOS_CATCH("");
}


void GetNormalAndReferencePlane()
{
KRATOS_TRY;

const auto& r_2d_model_part = this->Get2DModelPart();
const bool is_distributed = r_2d_model_part.IsDistributed();
auto p_geometry = GetGeometryFromModelPart(r_2d_model_part);

const auto& r_communicator = r_2d_model_part.GetCommunicator();
const auto& r_data_communicator = r_communicator.GetDataCommunicator();
const int mpi_rank = r_data_communicator.Rank();
const int mpi_size = r_data_communicator.Size();

const int partition_entity = DeterminePartitionWithEntities(r_2d_model_part);

const int tag_send_normal = 1;
const int tag_send_point = 2;

if (partition_entity != mpi_rank) {
if (is_distributed) {
r_data_communicator.Recv(mNormalPlane, partition_entity, tag_send_normal);
r_data_communicator.Recv(mPointPlane.Coordinates(), partition_entity, tag_send_point);
}
} else {
GeometryType::CoordinatesArrayType aux_coords;
noalias(mPointPlane.Coordinates()) = p_geometry->Center();
p_geometry->PointLocalCoordinates(aux_coords, mPointPlane);
noalias(mNormalPlane) = p_geometry->UnitNormal(aux_coords);

std::size_t check_normal;
const double numerical_limit = std::numeric_limits<double>::epsilon() * 1.0e4;
struct normal_check {
normal_check(array_1d<double, 3>& rNormal) : reference_normal(rNormal) {};
array_1d<double, 3> reference_normal;
GeometryType::CoordinatesArrayType aux_coords;
};
if (mEntityTypeMesh == EntityTypeMesh::CONDITIONS) {
check_normal = block_for_each<SumReduction<std::size_t>>(r_2d_model_part.Conditions(), normal_check(mNormalPlane), [&numerical_limit](auto& r_cond, normal_check& nc) {
auto& r_geom = r_cond.GetGeometry();
r_geom.PointLocalCoordinates(nc.aux_coords, r_geom.Center());
const auto normal = r_geom.UnitNormal(nc.aux_coords);
return (norm_2(normal - nc.reference_normal) > numerical_limit);
});
} else {
check_normal = block_for_each<SumReduction<std::size_t>>(r_2d_model_part.Elements(), normal_check(mNormalPlane), [&numerical_limit](auto& r_elem, normal_check& nc) {
auto& r_geom = r_elem.GetGeometry();
r_geom.PointLocalCoordinates(nc.aux_coords, r_geom.Center());
const auto normal = r_geom.UnitNormal(nc.aux_coords);
return (norm_2(normal - nc.reference_normal) > numerical_limit);
});
}
KRATOS_ERROR_IF_NOT(check_normal == 0) << "The 2D reference model part has not consistent normals. Please check that is properly aligned" << std::endl;

if (is_distributed) {
const auto& r_point_coordinates = mPointPlane.Coordinates();
for (int i_rank = 0; i_rank < mpi_size; ++i_rank) {
if (i_rank != partition_entity) {
r_data_communicator.Send(mNormalPlane, i_rank, tag_send_normal);
r_data_communicator.Send(r_point_coordinates, i_rank, tag_send_point);
}
}
}
}

KRATOS_CATCH("");
}


void CreateBaseMapper()
{
KRATOS_TRY;

auto& r_origin_model_part = this->GetOriginModelPart();
auto& r_destination_model_part = this->GetDestinationModelPart();

if (mMetaMapperType == "nearest_neighbor") {
if (mCopiedParameters.Has("interpolation_type")) mCopiedParameters.RemoveValue("interpolation_type");
if (mCopiedParameters.Has("local_coord_tolerance")) mCopiedParameters.RemoveValue("local_coord_tolerance");
mpBaseMapper = Kratos::make_unique<NearestNeighborMapperType>(r_origin_model_part, r_destination_model_part, mCopiedParameters);
} else if (mMetaMapperType == "nearest_element") {
if (mCopiedParameters.Has("interpolation_type")) mCopiedParameters.RemoveValue("interpolation_type");
mpBaseMapper = Kratos::make_unique<NearestElementMapperType>(r_origin_model_part, r_destination_model_part, mCopiedParameters);
} else if (mMetaMapperType == "barycentric") {
mpBaseMapper = Kratos::make_unique<BarycentricMapperType>(r_origin_model_part, r_destination_model_part, mCopiedParameters);
} else {
KRATOS_ERROR << "Mapper " << mCopiedParameters["base_mapper"].GetString() << " is not available as base mapper for projection" << std::endl;
}

KRATOS_CATCH("");
}


void MoveModelParts()
{
KRATOS_TRY;

auto& r_3d_model_part = this->Get3DModelPart();

MapperUtilities::SaveCurrentConfiguration(r_3d_model_part);

struct ProjectionVariables
{
ProjectionVariables(array_1d<double, 3>& rNormal, Point& rPoint) : reference_normal(rNormal), reference_point(rPoint) {};
array_1d<double, 3> reference_normal;
Point reference_point;
double distance;
array_1d<double, 3> projected_point_coordinates;
};

block_for_each(r_3d_model_part.Nodes(), ProjectionVariables(mNormalPlane, mPointPlane), [&](auto& r_node, ProjectionVariables& p) {
noalias(p.projected_point_coordinates) = GeometricalProjectionUtilities::FastProject(p.reference_point, r_node, p.reference_normal, p.distance).Coordinates();
noalias(r_node.Coordinates()) = p.projected_point_coordinates;
});

KRATOS_CATCH("");
}


void UnMoveModelParts()
{
KRATOS_TRY; 

auto& r_3d_model_part = this->Get3DModelPart();

MapperUtilities::RestoreCurrentConfiguration(r_3d_model_part);

KRATOS_CATCH("");
}

void CreateMapperLocalSystems(
const Communicator& rModelPartCommunicator,
std::vector<Kratos::unique_ptr<MapperLocalSystem>>& rLocalSystems
) override
{
AccessorInterpolativeMapperBase<TMapperBackend>::CreateMapperLocalSystems(*mpBaseMapper, rModelPartCommunicator, rLocalSystems);
}

MapperInterfaceInfoUniquePointerType GetMapperInterfaceInfo() const override
{
return AccessorInterpolativeMapperBase<TMapperBackend>::GetMapperInterfaceInfo(*mpBaseMapper);
}

Parameters GetMapperDefaultSettings() const override
{
return Parameters( R"({
"search_settings"                    : {},
"echo_level"                         : 0,
"interpolation_type"                 : "unspecified",
"local_coord_tolerance"              : 0.25,
"use_initial_configuration"          : false,
"print_pairing_status_to_file"       : false,
"pairing_status_file_path"           : "",
"base_mapper"                        : "nearest_neighbor"
})");
}


}; 

}  
