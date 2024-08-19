
#pragma once


#include <unordered_map>

#include "processes/process.h"
#include "includes/key_hash.h"
#include "includes/model_part.h"

namespace Kratos
{







class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) PrismNeighboursProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(PrismNeighboursProcess);

typedef Node                                          NodeType;
typedef Geometry<NodeType>                           GeometryType;

typedef ModelPart::NodesContainerType              NodesArrayType;
typedef ModelPart::ConditionsContainerType    ConditionsArrayType;
typedef ModelPart::ElementsContainerType        ElementsArrayType;

typedef NodesArrayType::iterator                NodesIterarorType;
typedef ConditionsArrayType::iterator      ConditionsIteratorType;
typedef ElementsArrayType::iterator          ElementsIteratorType;

typedef GlobalPointersVector<NodeType> NodePointerVector;
typedef GlobalPointersVector<Element> ElementPointerVector;

typedef std::size_t IndexType;

typedef std::size_t SizeType;

typedef vector<IndexType> VectorIndexType;

typedef VectorIndexHasher<VectorIndexType> VectorIndexHasherType;

typedef VectorIndexComparor<VectorIndexType> VectorIndexComparorType;

typedef std::unordered_map<VectorIndexType, IndexType, VectorIndexHasherType, VectorIndexComparorType > HashMapVectorIntIntType;

typedef HashMapVectorIntIntType::iterator HashMapVectorIntIntIteratorType;

typedef std::unordered_map<VectorIndexType, Element::Pointer, VectorIndexHasherType, VectorIndexComparorType > HashMapVectorIntElementPointerType;

typedef HashMapVectorIntElementPointerType::iterator HashMapVectorIntElementPointerIteratorType;



PrismNeighboursProcess(
ModelPart& rModelPart,
const bool ComputeOnNodes = false
) : mrModelPart(rModelPart),
mComputeOnNodes(ComputeOnNodes)
{
}

virtual ~PrismNeighboursProcess() {}


void operator()()
{
Execute();
}



void Execute() override;


void ExecuteInitialize() override;


void ExecuteFinalize() override;






virtual std::string Info() const override
{
return "PrismNeighboursProcess";
}

virtual void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "PrismNeighboursProcess";
}

virtual void PrintData(std::ostream& rOStream) const override
{
}



protected:








private:



ModelPart& mrModelPart;     
const bool mComputeOnNodes; 



void ClearNeighbours();


template< class TDataType >
void  AddUniqueWeakPointer(
GlobalPointersVector< TDataType >& rPointerVector,
const typename TDataType::WeakPointer Candidate
)
{
typename GlobalPointersVector< TDataType >::iterator beginit = rPointerVector.begin();
typename GlobalPointersVector< TDataType >::iterator endit   = rPointerVector.end();
while ( beginit != endit && beginit->Id() != (Candidate)->Id()) {
beginit++;
}
if( beginit == endit ) {
rPointerVector.push_back(Candidate);
}

}








PrismNeighboursProcess& operator=(PrismNeighboursProcess const& rOther);


}; 






inline std::istream& operator >> (std::istream& rIStream,
PrismNeighboursProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const PrismNeighboursProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  
