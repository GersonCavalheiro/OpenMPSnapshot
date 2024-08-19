
#pragma once

#include <string>
#include <iostream>



#include "includes/define.h"
#include "includes/model_part.h"
#include "containers/sparse_contiguous_row_graph.h"


namespace Kratos
{







class KRATOS_API(KRATOS_CORE) ModelPartGraphUtilities
{
public:
typedef unsigned int IndexType;

KRATOS_CLASS_POINTER_DEFINITION(ModelPartGraphUtilities);


ModelPartGraphUtilities() = delete;

ModelPartGraphUtilities(ModelPartGraphUtilities const& rOther) = delete;


ModelPartGraphUtilities& operator=(ModelPartGraphUtilities const& rOther) = delete;



static Kratos::unique_ptr<SparseContiguousRowGraph<>> ComputeGraph(const ModelPart& rModelPart);


static std::pair<DenseVector<IndexType>, DenseVector<IndexType>> ComputeCSRGraph(const ModelPart& rModelPart);



static std::pair<IndexType, DenseVector<double>> ComputeConnectedComponents(
const ModelPart::NodesContainerType& rNodes,
const DenseVector<IndexType>& rRowIndices,
const DenseVector<IndexType>& rColIndices
);

static std::pair<IndexType, DenseVector<double>> ComputeConnectedComponentsWithActiveNodesCheck(
const ModelPart::NodesContainerType& rNodes,
const DenseVector<IndexType>& rRowIndices,
const DenseVector<IndexType>& rColIndices,
const std::vector<bool>& active_nodes_list
);

static std::vector<IndexType> ApplyMinimalScalarFixity(
ModelPart::NodesContainerType& rNodes,
const Variable<double>& rVar,
const DenseVector<double>& colors,
const IndexType ncolors
);









protected:






static void BreadthFirstSearch(
const int startVertex,
const int color,
const DenseVector<IndexType>& rRowIndices,
const DenseVector<IndexType>& rColIndices,
std::unordered_map<IndexType, int>& rVisited);

static void BreadthFirstSearchWithActiveNodesCheck(
const int startVertex,
const int color,
const DenseVector<IndexType>& rRowIndices,
const DenseVector<IndexType>& rColIndices,
std::unordered_map<IndexType, int>& rVisited,
const std::unordered_map<IndexType, bool>& rActiveNodes);








private:














}; 







}  



