
#pragma once





#include "wave_element.h"

namespace Kratos
{






template<std::size_t TNumNodes>
class PrimitiveElement : public WaveElement<TNumNodes>
{
public:

typedef std::size_t IndexType;

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;

typedef WaveElement<TNumNodes> BaseType;

typedef typename BaseType::NodesArrayType NodesArrayType;

typedef typename BaseType::PropertiesType PropertiesType;

typedef typename BaseType::ElementData ElementData;

typedef typename BaseType::LocalMatrixType LocalMatrixType;

typedef typename BaseType::LocalVectorType LocalVectorType;

typedef typename BaseType::ShapeFunctionsGradientsType ShapeFunctionsGradientsType;


KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(PrimitiveElement);



PrimitiveElement() : BaseType() {}


PrimitiveElement(IndexType NewId, const NodesArrayType& ThisNodes) : BaseType(NewId, ThisNodes) {}


PrimitiveElement(IndexType NewId, GeometryType::Pointer pGeometry) : BaseType(NewId, pGeometry) {}


PrimitiveElement(IndexType NewId, GeometryType::Pointer pGeometry, typename PropertiesType::Pointer pProperties) : BaseType(NewId, pGeometry, pProperties) {}


~ PrimitiveElement() override {};



Element::Pointer Create(IndexType NewId, NodesArrayType const& ThisNodes, typename PropertiesType::Pointer pProperties) const override
{
return Kratos::make_intrusive<PrimitiveElement<TNumNodes>>(NewId, this->GetGeometry().Create(ThisNodes), pProperties);
}


Element::Pointer Create(IndexType NewId, GeometryType::Pointer pGeom, typename PropertiesType::Pointer pProperties) const override
{
return Kratos::make_intrusive<PrimitiveElement<TNumNodes>>(NewId, pGeom, pProperties);
}


Element::Pointer Clone(IndexType NewId, NodesArrayType const& ThisNodes) const override
{
Element::Pointer p_new_elem = Create(NewId, this->GetGeometry().Create(ThisNodes), this->pGetProperties());
p_new_elem->SetData(this->GetData());
p_new_elem->Set(Flags(*this));
return p_new_elem;
}



std::string Info() const override
{
return "PrimitiveElement";
}


protected:

static constexpr IndexType mLocalSize = BaseType::mLocalSize;


void UpdateGaussPointData(
ElementData& rData,
const array_1d<double,TNumNodes>& rN) override;

double StabilizationParameter(const ElementData& rData) const override;


private:

friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, Element);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, Element);
}






}; 





}  
