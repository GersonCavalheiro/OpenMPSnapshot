
#pragma once



#include "includes/define.h"
#include "containers/array_1d.h"
#include "includes/serializer.h"

namespace Kratos
{







class Point : public array_1d<double, 3>
{
static constexpr std::size_t mDimension = 3;

public:


KRATOS_CLASS_POINTER_DEFINITION(Point);

typedef array_1d<double, mDimension> BaseType;

typedef BaseType CoordinatesArrayType;

typedef std::size_t SizeType;

typedef std::size_t IndexType;


Point() : BaseType()
{
SetAllCoordinates();
}

Point(double NewX, double NewY = 0, double NewZ = 0) : BaseType()
{
this->operator()(0) = NewX;
this->operator()(1) = NewY;
this->operator()(2) = NewZ;
}


Point(Point const &rOtherPoint)
: BaseType(rOtherPoint) {}


explicit Point(CoordinatesArrayType const &rOtherCoordinates)
: BaseType(rOtherCoordinates) {}


template <class TVectorType>
explicit Point(vector_expression<TVectorType> const &rOtherCoordinates)
: BaseType(rOtherCoordinates) {}


explicit Point(std::vector<double> const &rOtherCoordinates) : BaseType()
{
SizeType size = rOtherCoordinates.size();
size = (mDimension < size) ? mDimension : size;
for (IndexType i = 0; i < size; i++)
this->operator[](i) = rOtherCoordinates[i];
}

virtual ~Point() {}


Point &operator=(const Point &rOther)
{
CoordinatesArrayType::operator=(rOther);
return *this;
}

bool operator==(const Point &rOther) const
{
return std::equal(this->begin(), this->end(), rOther.begin());
}



double SquaredDistance(const Point& rOtherPoint) const
{
const array_1d<double, 3> diff_vector = this->Coordinates() - rOtherPoint.Coordinates();
return (std::pow(diff_vector[0], 2) + std::pow(diff_vector[1], 2) + std::pow(diff_vector[2], 2));
}


double Distance(const Point& rOtherPoint) const
{
return norm_2(this->Coordinates() - rOtherPoint.Coordinates());
}


static constexpr IndexType Dimension()
{
return mDimension;
}


double X() const
{
return this->operator[](0);
}


double Y() const
{
return this->operator[](1);
}


double Z() const
{
return this->operator[](2);
}

double &X()
{
return this->operator[](0);
}


double &Y()
{
return this->operator[](1);
}


double &Z()
{
return this->operator[](2);
}

CoordinatesArrayType const &Coordinates() const
{
return *this;
}

CoordinatesArrayType &Coordinates()
{
return *this;
}


virtual std::string Info() const
{
return "Point";
}

virtual void PrintInfo(std::ostream &rOStream) const
{
rOStream << this->Info();
}

virtual void PrintData(std::ostream &rOStream) const
{
rOStream << " ("  << this->operator[](0)
<< ", " << this->operator[](1)
<< ", " << this->operator[](2)
<< ")";
}


private:

void SetAllCoordinates(double const &Value = double())
{
for (IndexType i = 0; i < mDimension; i++)
this->operator()(i) = Value;
}


friend class Serializer;

virtual void save(Serializer &rSerializer) const
{
rSerializer.save_base("BaseClass", *static_cast<const array_1d<double, mDimension> *>(this));
}

virtual void load(Serializer &rSerializer)
{
rSerializer.load_base("BaseClass", *static_cast<array_1d<double, mDimension> *>(this));
}


}; 




inline std::istream &operator>>(std::istream &rIStream,
Point &rThis){
return rIStream;
}

inline std::ostream &operator<<(std::ostream &rOStream,
const Point &rThis)
{
rThis.PrintInfo(rOStream);
rThis.PrintData(rOStream);

return rOStream;
}

} 
