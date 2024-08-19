
#pragma once


#include "includes/define.h"
#include "spaces/ublas_space.h"
#include "spatial_containers/spatial_containers.h"
#include "containers/container_expression/container_expression.h"

#include "custom_utilities/collective_expressions.h"

namespace Kratos
{


class KRATOS_API(OPTIMIZATION_APPLICATION) ContainerExpressionUtils
{
public:

using IndexType = std::size_t;

using SparseSpaceType = UblasSpace<double, CompressedMatrix, Vector>;

using SparseMatrixType = SparseSpaceType::MatrixType;



template<class TContainerType>
static double NormInf(const ContainerExpression<TContainerType>& rContainer);


static double NormInf(const CollectiveExpressions& rContainer);


template<class TContainerType>
static double NormL2(const ContainerExpression<TContainerType>& rContainer);


static double NormL2(const CollectiveExpressions& rContainer);


template<class TContainerType>
static double EntityMaxNormL2(const ContainerExpression<TContainerType>& rContainer);


template<class TContainerType>
static double InnerProduct(
const ContainerExpression<TContainerType>& rContainer1,
const ContainerExpression<TContainerType>& rContainer2);


static double InnerProduct(
const CollectiveExpressions& rContainer1,
const CollectiveExpressions& rContainer2);


template<class TContainerType>
static void ProductWithEntityMatrix(
ContainerExpression<TContainerType>& rOutput,
const SparseMatrixType& rMatrix,
const ContainerExpression<TContainerType>& rInput);


template<class TContainerType>
static void ProductWithEntityMatrix(
ContainerExpression<TContainerType>& rOutput,
const Matrix& rMatrix,
const ContainerExpression<TContainerType>& rInput);


static void Transpose(
Matrix& rOutput,
const Matrix& rInput);


static void Transpose(
SparseMatrixType& rOutput,
const SparseMatrixType& rInput);


template<class TContainerType>
static void ComputeNumberOfNeighbourEntities(
ContainerExpression<ModelPart::NodesContainerType>& rOutput);


template<class TContainerType>
static void MapContainerVariableToNodalVariable(
ContainerExpression<ModelPart::NodesContainerType>& rOutput,
const ContainerExpression<TContainerType>& rInput,
const ContainerExpression<ModelPart::NodesContainerType>& rNeighbourEntities);


template<class TContainerType>
static void MapNodalVariableToContainerVariable(
ContainerExpression<TContainerType>& rOutput,
const ContainerExpression<ModelPart::NodesContainerType>& rInput);


template<class TContainerType>
static void ComputeNodalVariableProductWithEntityMatrix(
ContainerExpression<ModelPart::NodesContainerType>& rOutput,
const ContainerExpression<ModelPart::NodesContainerType>& rNodalValues,
const Variable<Matrix>& rMatrixVariable,
TContainerType& rEntities);

};

}