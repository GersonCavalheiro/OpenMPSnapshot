


#include "custom_python/python_outfitted_constitutive_law.hpp"

namespace Kratos
{

/
Matrix& PythonOutfittedConstitutiveLaw::Transform2DTo3D (Matrix& rMatrix)
{


if (rMatrix.size1() == 2 && rMatrix.size2() == 2)
{

rMatrix.resize( 3, 3, true);

rMatrix( 0 , 2 ) = 0.0;
rMatrix( 1 , 2 ) = 0.0;

rMatrix( 2 , 0 ) = 0.0;
rMatrix( 2 , 1 ) = 0.0;

rMatrix( 2 , 2 ) = 1.0;

}
else if(rMatrix.size1() != 3 && rMatrix.size2() != 3)
{

KRATOS_ERROR << "Matrix Dimensions are not correct" << std::endl;

}

return rMatrix;
}



void PythonOutfittedConstitutiveLaw::GetLawFeatures(Features& rFeatures)
{
boost::python::call_method<void>(mpPyConstitutiveLaw->ptr(), "GetLawFeatures", boost::ref<Features>(rFeatures));
}


int PythonOutfittedConstitutiveLaw::Check(const Properties& rProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo)
{
return boost::python::call_method<int>(mpPyConstitutiveLaw->ptr(),"Check", boost::ref<const Properties>(rProperties),boost::ref<const GeometryType>(rElementGeometry),boost::ref<const ProcessInfo>(rCurrentProcessInfo));
}

} 
