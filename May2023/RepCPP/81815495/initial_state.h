
# pragma once

#include <atomic>


#include "includes/define.h"
#include "includes/variables.h"

namespace Kratos
{



class KRATOS_API(KRATOS_CORE) InitialState
{
public:

using SizeType = std::size_t;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(InitialState);


enum class InitialImposingType
{
STRAIN_ONLY = 0,
STRESS_ONLY = 1,
DEFORMATION_GRADIENT_ONLY = 2,
STRAIN_AND_STRESS = 3,
DEFORMATION_GRADIENT_AND_STRESS = 4
};


InitialState()
{}

InitialState(const SizeType Dimension);

InitialState(const Vector& rInitialStrainVector,
const Vector& rInitialStressVector,
const Matrix& rInitialDeformationGradientMatrix);

InitialState(const Vector& rImposingEntity,
const InitialImposingType InitialImposition = InitialImposingType::STRAIN_ONLY);

InitialState(const Vector& rInitialStrainVector,
const Vector& rInitialStressVector);

InitialState(const Matrix& rInitialDeformationGradientMatrix);

virtual ~InitialState() {}



/
void SetInitialStrainVector(const Vector& rInitialStrainVector);


void SetInitialStressVector(const Vector& rInitialStressVector);


void SetInitialDeformationGradientMatrix(const Matrix& rInitialDeformationGradientMatrix);


const Vector& GetInitialStrainVector() const;


const Vector& GetInitialStressVector() const;


const Matrix& GetInitialDeformationGradientMatrix() const;



virtual std::string Info() const
{
std::stringstream buffer;
buffer << "InitialState" ;

return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const  {rOStream << "InitialState";}

virtual void PrintData(std::ostream& rOStream) const {}


private:

Vector mInitialStrainVector;
Vector mInitialStressVector;
Matrix mInitialDeformationGradientMatrix;


mutable std::atomic<int> mReferenceCounter{0};
friend void intrusive_ptr_add_ref(const InitialState* x)
{
x->mReferenceCounter.fetch_add(1, std::memory_order_relaxed);
}

friend void intrusive_ptr_release(const InitialState* x)
{
if (x->mReferenceCounter.fetch_sub(1, std::memory_order_release) == 1) {
std::atomic_thread_fence(std::memory_order_acquire);
delete x;
}
}


friend class Serializer;

void save(Serializer& rSerializer) const
{
rSerializer.save("InitialStrainVector",mInitialStrainVector);
rSerializer.save("InitialStressVector",mInitialStressVector);
rSerializer.save("InitialDeformationGradientMatrix",mInitialDeformationGradientMatrix);
}

void load(Serializer& rSerializer)
{
rSerializer.load("InitialStrainVector",mInitialStrainVector);
rSerializer.load("InitialStressVector",mInitialStressVector);
rSerializer.load("InitialDeformationGradientMatrix",mInitialDeformationGradientMatrix);
}


}; 


} 
