
#pragma once






#include "includes/define.h"
#include "utilities/math_utils.h"


namespace Kratos
{





namespace StaticCondensationUtility
{
typedef Element ElementType;
typedef std::size_t SizeType;
typedef Matrix MatrixType;



void CondenseLeftHandSide(
ElementType& rTheElement,
MatrixType& rLeftHandSideMatrix,
const std::vector<int> & rDofList);



std::vector<MatrixType> CalculateSchurComplements(
ElementType& rTheElement,
const MatrixType& rLeftHandSideMatrix,
const std::vector<int> & rDofList);



std::vector<int> CreateRemainingDofList(
ElementType& rTheElement,
const std::vector<int> & rDofList);



void FillSchurComplements(
MatrixType& Submatrix,
const MatrixType& rLeftHandSideMatrix,
const std::vector<int>& rVecA,
const std::vector<int>& rVecB,
const SizeType& rSizeA,
const SizeType& rSizeB); 



void ConvertingCondensation(
ElementType& rTheElement,
Vector& rLocalizedDofVector,
Vector& rValues,
const std::vector<int>& rDofList,
const MatrixType& rLeftHandSideMatrix);



SizeType GetNumDofsElement(const ElementType& rTheElement);

}  

}  


