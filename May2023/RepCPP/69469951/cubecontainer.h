


#pragma once


#include <map>

template<typename Dimension1, typename Dimension2, typename Dimension3, typename T>
class CubeContainer
{
public:
struct Index
{
Dimension1 dim1;
Dimension2 dim2;
Dimension3 dim3;

Index( Dimension1 whichDim1, Dimension2 whichDim2, Dimension3 whichDim3 );
bool operator<( const Index& whichIndex ) const;
};

typedef typename std::map<Index, T>::iterator iterator;
typedef typename std::map<Index, T>::iterator const_iterator;

T& operator()( const Dimension1& dim1, const Dimension2& dim2, const Dimension3& dim3 );
const T& operator()( const Dimension1& dim1, const Dimension2& dim2, const Dimension3& dim3 ) const;

size_t size() const;
typename CubeContainer::const_iterator find( const Dimension1& dim1, const Dimension2& dim2, const Dimension3& dim3 ) const;
typename CubeContainer::iterator find( const Dimension1& dim1, const Dimension2& dim2, const Dimension3& dim3 );

typename CubeContainer::iterator begin();
typename CubeContainer::iterator end();

private:
std::map<Index, T> cube;

};

#include "cubecontainer_impl.h"


