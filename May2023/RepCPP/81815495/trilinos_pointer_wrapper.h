
#pragma once



#include "trilinos_space.h"

namespace Kratos
{




typedef TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector> TrilinosSparseSpaceType;





class AuxiliaryMatrixWrapper
{
public:

typedef typename TrilinosSparseSpaceType::MatrixType TrilinosMatrixType;
typedef typename TrilinosSparseSpaceType::MatrixPointerType TrilinosMatrixPointerType;


AuxiliaryMatrixWrapper(TrilinosMatrixPointerType p) : mp(p){};

virtual ~AuxiliaryMatrixWrapper(){}




TrilinosMatrixPointerType& GetPointer() { return mp; }


TrilinosMatrixType& GetReference() { return *mp; }




virtual std::string Info() const {
std::stringstream buffer;
buffer << "AuxiliaryMatrixWrapper" ;
return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const {rOStream << "AuxiliaryMatrixWrapper";}

virtual void PrintData(std::ostream& rOStream) const {}

private:



TrilinosMatrixPointerType mp;






AuxiliaryMatrixWrapper& operator=(const AuxiliaryMatrixWrapper &rOther) = delete;

}; 


class AuxiliaryVectorWrapper
{
public:

typedef typename TrilinosSparseSpaceType::VectorType TrilinosVectorType;
typedef typename TrilinosSparseSpaceType::VectorPointerType TrilinosVectorPointerType;


AuxiliaryVectorWrapper(TrilinosVectorPointerType p) : mp(p){};

virtual ~AuxiliaryVectorWrapper(){}




TrilinosVectorPointerType& GetPointer() { return mp; }


TrilinosVectorType& GetReference() { return *mp; }




virtual std::string Info() const {
std::stringstream buffer;
buffer << "AuxiliaryVectorWrapper" ;
return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const {rOStream << "AuxiliaryVectorWrapper";}

virtual void PrintData(std::ostream& rOStream) const {}

private:



TrilinosVectorPointerType mp;






AuxiliaryVectorWrapper& operator=(AuxiliaryVectorWrapper const& rOther) = delete;

}; 

} 