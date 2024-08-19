
#if !defined(KRATOS_ELEMENTAL_NEIGHBOURS_SEARCH_PROCESS_H_INCLUDED )
#define  KRATOS_ELEMENTAL_NEIGHBOURS_SEARCH_PROCESS_H_INCLUDED



#include "processes/process.h"
#include "includes/node.h"
#include "includes/element.h"
#include "includes/model_part.h"
#include "utilities/openmp_utils.h"
#include "custom_processes/mesher_process.hpp"
#include "delaunay_meshing_application_variables.h"

namespace Kratos
{


class ElementalNeighboursSearchProcess
: public MesherProcess
{
public:
KRATOS_CLASS_POINTER_DEFINITION( ElementalNeighboursSearchProcess );

typedef  ModelPart::NodesContainerType NodesContainerType;
typedef  ModelPart::ElementsContainerType ElementsContainerType;

typedef Node::WeakPointer NodeWeakPtrType;
typedef Element::WeakPointer ElementWeakPtrType;
typedef Condition::WeakPointer ConditionWeakPtrType;

typedef GlobalPointersVector<Node > NodeWeakPtrVectorType;
typedef GlobalPointersVector<Element> ElementWeakPtrVectorType;
typedef GlobalPointersVector<Condition> ConditionWeakPtrVectorType;


ElementalNeighboursSearchProcess(ModelPart& rModelPart,
int Dimension,
int EchoLevel = 0,
int AverageElements = 10)
: mrModelPart(rModelPart)
{
mAverageElements = AverageElements;
mDimension       = Dimension;
mEchoLevel       = EchoLevel;
}

virtual ~ElementalNeighboursSearchProcess()
{
}



void operator()()
{
Execute();
}



void Execute() override
{
bool success=false;

int method = 0;  


if(method==0)
{
success=KratosSearch();
}
else
{
success=LohnerSearch(); 
}

if(!success)
{
std::cout<<" ERROR:  Element Neighbours Search FAILED !!! "<<std::endl;
}


};


void ClearNeighbours()
{
for(auto& i_node: mrModelPart.Nodes())
{
ElementWeakPtrVectorType& nElements = i_node.GetValue(NEIGHBOUR_ELEMENTS);
nElements.clear();
}
for(auto& i_elem : mrModelPart.Elements())
{
ElementWeakPtrVectorType& nElements = i_elem.GetValue(NEIGHBOUR_ELEMENTS);
nElements.clear();
}
}






std::string Info() const override
{
return "ElementalNeighboursSearchProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ElementalNeighboursSearchProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:

private:
ModelPart& mrModelPart;
int mAverageElements;
int mDimension;
int mEchoLevel;

template<class TDataType> void  AddUniquePointer
(GlobalPointersVector<TDataType>& v, const typename TDataType::WeakPointer candidate)
{
typename GlobalPointersVector< TDataType >::iterator i = v.begin();
typename GlobalPointersVector< TDataType >::iterator endit = v.end();
while ( i != endit && (i)->Id() != (candidate.lock())->Id())
{
i++;
}
if( i == endit )
{
v.push_back(candidate);
}

}

ElementWeakPtrType CheckForNeighbourElems1D (unsigned int Id_1, ElementWeakPtrVectorType& nElements, ElementsContainerType::iterator i_elem)
{
for(auto i_nelem(nElements.begin()); i_nelem != nElements.end(); ++i_nelem)
{
Geometry<Node >& nGeometry = i_nelem->GetGeometry();
if(nGeometry.LocalSpaceDimension() == 1){
for(unsigned int node_i = 0; node_i < nGeometry.size(); ++node_i)
{
if(nGeometry[node_i].Id() == Id_1)
{
if(i_nelem->Id() != i_elem->Id())
{
return Element::WeakPointer(*i_nelem.base());
}
}
}
}
}
return Element::WeakPointer(*i_elem.base());
}


ElementWeakPtrType CheckForNeighbourElems2D (unsigned int Id_1, unsigned int Id_2, ElementWeakPtrVectorType& nElements, ElementsContainerType::iterator i_elem)
{
for(auto i_nelem(nElements.begin()); i_nelem != nElements.end(); ++i_nelem)
{
Geometry<Node >& nGeometry = i_nelem->GetGeometry();
if(nGeometry.LocalSpaceDimension() == 2){
for(unsigned int node_i = 0; node_i < nGeometry.size(); ++node_i)
{
if (nGeometry[node_i].Id() == Id_2)
{
if(i_nelem->Id() != i_elem->Id())
{
return *i_nelem.base();
}
}
}
}
}
return *i_elem.base();
}

ElementWeakPtrType CheckForNeighbourElems3D (unsigned int Id_1, unsigned int Id_2, unsigned int Id_3, ElementWeakPtrVectorType& nElements, ElementsContainerType::iterator i_elem)
{
for(auto i_nelem(nElements.begin()); i_nelem != nElements.end(); ++i_nelem)
{
Geometry<Node >& nGeometry = i_nelem->GetGeometry();
if(nGeometry.LocalSpaceDimension() == 3){
for(unsigned int node_i = 0; node_i < nGeometry.size(); ++node_i)
{
if(nGeometry[node_i].Id() == Id_2)
{
for(unsigned int node_j = 0; node_j < nGeometry.size(); ++node_j)
{
if (nGeometry[node_j].Id() == Id_3)
if(i_nelem->Id() != i_elem->Id())
{
return *i_nelem.base();
}
}
}
}
}
}
return *i_elem.base();
}


void ResetFlagOptions (Node& rNode)
{
rNode.Reset(BOUNDARY);
}

void ResetFlagOptions (Element& rElement)
{
rElement.Reset(BOUNDARY);
}


void CleanElementNeighbours()
{

KRATOS_TRY


for(auto& i_node : mrModelPart.Nodes())
{
auto& nElements = i_node.GetValue(NEIGHBOUR_ELEMENTS);
nElements.clear();
nElements.reserve(mAverageElements);

ResetFlagOptions(i_node);
}

for(auto& i_elem : mrModelPart.Elements())
{
auto& nElements = i_elem.GetValue(NEIGHBOUR_ELEMENTS);
nElements.clear();
nElements.resize(i_elem.GetGeometry().FacesNumber());

ResetFlagOptions(i_elem);
}

KRATOS_CATCH( "" )
}


void PrintElementNeighbours()
{
KRATOS_TRY

std::cout<<" NODES: neighbour elems: "<<std::endl;
for(auto& i_node : mrModelPart.Nodes())
{
std::cout<<"["<<i_node.Id()<<"]:"<<std::endl;
std::cout<<"( ";
auto& nElements = i_node.GetValue(NEIGHBOUR_ELEMENTS);
for(const auto& i_nelem : nElements)
{
std::cout<< i_nelem.Id()<<", ";
}
std::cout<<" )"<<std::endl;
}

std::cout<<std::endl;

std::cout<<" ELEMENTS: neighbour elems: "<<std::endl;

for(auto& i_elem : mrModelPart.Elements())
{
std::cout<<"["<<i_elem.Id()<<"]:"<<std::endl;
std::cout<<"( ";
auto& nElements = i_elem.GetValue(NEIGHBOUR_ELEMENTS);
for(auto& i_nelem : nElements)
{
std::cout<< i_nelem.Id()<<", ";
}
std::cout<<" )"<<std::endl;
}


std::cout<<std::endl;

KRATOS_CATCH( "" )
}



bool KratosSearch()
{

KRATOS_TRY

ElementsContainerType& rElements = mrModelPart.Elements();


CleanElementNeighbours();


for(auto i_elem(rElements.begin()); i_elem != rElements.end(); ++i_elem)
{
Element::GeometryType& rGeometry = i_elem->GetGeometry();
for(unsigned int i = 0; i < rGeometry.size(); ++i)
{
rGeometry[i].GetValue(NEIGHBOUR_ELEMENTS).push_back(*i_elem.base());
}
}


unsigned int search_performed = false;

if (mDimension==2)
{
for(auto i_elem(rElements.begin()); i_elem != rElements.end(); ++i_elem)
{
Geometry<Node >& rGeometry = i_elem->GetGeometry();

if( rGeometry.FacesNumber() == 3 ){

auto& nElements = i_elem->GetValue(NEIGHBOUR_ELEMENTS);
if(nElements.size() != 3 )
nElements.resize(3);


nElements(0) = CheckForNeighbourElems2D(rGeometry[1].Id(), rGeometry[2].Id(), rGeometry[1].GetValue(NEIGHBOUR_ELEMENTS), i_elem);
nElements(1) = CheckForNeighbourElems2D(rGeometry[2].Id(), rGeometry[0].Id(), rGeometry[2].GetValue(NEIGHBOUR_ELEMENTS), i_elem);
nElements(2) = CheckForNeighbourElems2D(rGeometry[0].Id(), rGeometry[1].Id(), rGeometry[0].GetValue(NEIGHBOUR_ELEMENTS), i_elem);

unsigned int iface=0;
for(auto& i_nelem : nElements)
{
if (i_nelem.Id() == i_elem->Id())  
{
i_elem->Set(BOUNDARY);

DenseMatrix<unsigned int> lpofa; 
rGeometry.NodesInFaces(lpofa);

for(unsigned int i = 1; i < rGeometry.FacesNumber(); ++i)
{
rGeometry[lpofa(i,iface)].Set(BOUNDARY);  
}
}
iface++;
}

}
else if( rGeometry.FacesNumber() == 2 ){

auto& nElements = i_elem->GetValue(NEIGHBOUR_ELEMENTS);

if( nElements.size() != 2 )
nElements.resize(2);


nElements(0) = CheckForNeighbourElems1D(rGeometry[0].Id(), rGeometry[0].GetValue(NEIGHBOUR_ELEMENTS), i_elem);
nElements(1) = CheckForNeighbourElems1D(rGeometry[1].Id(), rGeometry[1].GetValue(NEIGHBOUR_ELEMENTS), i_elem);

unsigned int iface=0;
for(auto& i_nelem : nElements)
{
if(i_nelem.Id() == i_elem->Id())  
{
i_elem->Set(BOUNDARY);

DenseMatrix<unsigned int> lpofa; 
rGeometry.NodesInFaces(lpofa);

for(unsigned int i = 1; i < rGeometry.FacesNumber(); ++i)
{
rGeometry[lpofa(i,iface)].Set(BOUNDARY);  
}
}
iface++;
}
}
}

search_performed = true;
}

if (mDimension==3)
{
for(auto i_elem(rElements.begin()); i_elem != rElements.end(); ++i_elem)
{
Geometry<Node >& rGeometry = i_elem->GetGeometry();

if(rGeometry.FacesNumber() == 4){

auto& nElements = i_elem->GetValue(NEIGHBOUR_ELEMENTS);

if(nElements.size() != 4)
nElements.resize(4);


nElements(0) = CheckForNeighbourElems3D(rGeometry[1].Id(), rGeometry[2].Id(), rGeometry[3].Id(), rGeometry[1].GetValue(NEIGHBOUR_ELEMENTS), i_elem);
nElements(1) = CheckForNeighbourElems3D(rGeometry[2].Id(), rGeometry[3].Id(), rGeometry[0].Id(), rGeometry[2].GetValue(NEIGHBOUR_ELEMENTS), i_elem);
nElements(2) = CheckForNeighbourElems3D(rGeometry[3].Id(), rGeometry[0].Id(), rGeometry[1].Id(), rGeometry[3].GetValue(NEIGHBOUR_ELEMENTS), i_elem);
nElements(3) = CheckForNeighbourElems3D(rGeometry[0].Id(), rGeometry[1].Id(), rGeometry[2].Id(), rGeometry[0].GetValue(NEIGHBOUR_ELEMENTS), i_elem);


unsigned int iface=0;
for(auto& i_nelem : nElements)
{
if(i_nelem.Id() == i_elem->Id())  
{
i_elem->Set(BOUNDARY);

DenseMatrix<unsigned int> lpofa; 
rGeometry.NodesInFaces(lpofa);

for(unsigned int i = 1; i < rGeometry.FacesNumber(); ++i)
{
rGeometry[lpofa(i,iface)].Set(BOUNDARY);  
}
}
iface++;
}

}
else if(rGeometry.FacesNumber() == 3){

auto& nElements = i_elem->GetValue(NEIGHBOUR_ELEMENTS);

if(nElements.size() != 3)
nElements.resize(3);


nElements(0) = CheckForNeighbourElems2D(rGeometry[1].Id(), rGeometry[2].Id(), rGeometry[1].GetValue(NEIGHBOUR_ELEMENTS), i_elem);
nElements(1) = CheckForNeighbourElems2D(rGeometry[2].Id(), rGeometry[0].Id(), rGeometry[2].GetValue(NEIGHBOUR_ELEMENTS), i_elem);
nElements(2) = CheckForNeighbourElems2D(rGeometry[0].Id(), rGeometry[1].Id(), rGeometry[0].GetValue(NEIGHBOUR_ELEMENTS), i_elem);

unsigned int iface=0;
for(auto& i_nelem : nElements)
{
if(i_nelem.Id() == i_elem->Id())  
{
i_elem->Set(BOUNDARY);

Geometry<Node >& rGeometry = (i_elem)->GetGeometry();

DenseMatrix<unsigned int> lpofa; 
rGeometry.NodesInFaces(lpofa);

for(unsigned int i = 1; i < rGeometry.FacesNumber(); ++i)
{
rGeometry[lpofa(i,iface)].Set(BOUNDARY);  
}
}
iface++;
}

}
}
search_performed = true;
}

if( mrModelPart.NumberOfElements()>0 && search_performed )
return true;
else
return false;

KRATOS_CATCH( "" )
}


bool LohnerSearch()
{

KRATOS_TRY

NodesContainerType& rNodes = mrModelPart.Nodes();
ElementsContainerType& rElements = mrModelPart.Elements();


unsigned int Ne=rElements.size();
unsigned int Np=rNodes.size();

CleanElementNeighbours();


for(auto i_elem(rElements.begin()); i_elem != rElements.end(); ++i_elem)
{
Element::GeometryType& rGeometry = i_elem->GetGeometry();
if(rGeometry.LocalSpaceDimension() == mrModelPart.GetProcessInfo()[SPACE_DIMENSION]){
for(unsigned int i = 0; i < rGeometry.size(); ++i)
{
rGeometry[i].GetValue(NEIGHBOUR_ELEMENTS).push_back(*i_elem.base());
}
}
}


unsigned int ipoin=0;
unsigned int nnofa=0;
unsigned int jelem=0;
unsigned int icoun=0;
unsigned int jpoin=0;
unsigned int nnofj=0;
unsigned int nface=0;

DenseVector<unsigned int> lnofa; 
DenseMatrix<unsigned int> lpofa; 

Element::GeometryType& rGeometry = rElements.begin()->GetGeometry(); 
unsigned int Nf= rGeometry.FacesNumber();     

rGeometry.NumberNodesInFaces(lnofa);
rGeometry.NodesInFaces(lpofa);

DenseVector<unsigned int> lhelp (Nf-1); 
lhelp.clear();
DenseVector<unsigned int> lpoin (Np+1);
lpoin.clear();


int el;
#pragma omp parallel for reduction(+:nface) private(el,ipoin,nnofa,jelem,icoun,jpoin,nnofj) firstprivate(lhelp,lpoin)
for (el=1; el<(int)Ne+1; ++el) 
{

for (unsigned int nf=0; nf<Nf; ++nf) 
{
nnofa=lnofa(nf);

rElements[el].GetValue(NEIGHBOUR_ELEMENTS)(nf) = rElements(el);

for (unsigned int t=0; t<nnofa; ++t)
{
lhelp(t)=rElements[el].GetGeometry()[lpofa(t,nf)].Id();  
lpoin(lhelp(t))=1;                                    
}

ipoin=lhelp(1);   

auto& nElements = rNodes[ipoin].GetValue(NEIGHBOUR_ELEMENTS);

for(auto& i_nelem : nElements)  
{
jelem=i_nelem.Id();
unsigned int ielem =rElements[el].Id();

if(jelem!=ielem)
{

for(unsigned int fel=0; fel<Nf; ++fel) 
{
nnofj=lnofa(fel);

if (nnofj==nnofa)
{

icoun=0;
for (unsigned int jnofa=0; jnofa<nnofa; ++jnofa) 
{
jpoin= rElements[jelem].GetGeometry()[lpofa(jnofa,fel)].Id();
icoun= icoun+lpoin(jpoin);
}

if(icoun==nnofa)
{
rElements[el].GetValue(NEIGHBOUR_ELEMENTS)(nf) = rElements(jelem);
}
}
}
}
}


if (rElements[el].GetValue(NEIGHBOUR_ELEMENTS)[nf].Id() == rElements[el].Id())  
{

rElements[el].Set(BOUNDARY);

for (unsigned int t=0; t<nnofa; ++t) 
{
rNodes[lhelp(t)].Set(BOUNDARY);  
}

nface+=1;

}


for (unsigned int r=0; r<nnofa; ++r)
{
lpoin(lhelp(r))=0;                            
}
}
}



for(auto& i_elem : rElements)
{
Element::GeometryType& rGeometry = i_elem.GetGeometry();
for(unsigned int i = 0; i < rGeometry.size(); ++i)
{
if(rGeometry[i].Is(BOUNDARY))
{
i_elem.Set(BOUNDARY);
}
}

}

return true;

KRATOS_CATCH( "" )
}








ElementalNeighboursSearchProcess& operator=(ElementalNeighboursSearchProcess const& rOther);




}; 






inline std::istream& operator >> (std::istream& rIStream,
ElementalNeighboursSearchProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const ElementalNeighboursSearchProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
