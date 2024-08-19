

#ifndef VCG_SPACE_INDEX_PERFECT_SPATIAL_HASHING_H
#define VCG_SPACE_INDEX_PERFECT_SPATIAL_HASHING_H

#pragma warning(disable : 4996)

#define _USE_GRID_UTIL_PARTIONING_ 1
#define _USE_OCTREE_PARTITIONING_  (1-_USE_GRID_UTIL_PARTIONING_)

#include <vector>
#include <list>
#include <algorithm>

#include <vcg/space/index/base.h>
#include <vcg/space/index/grid_util.h>

#include <vcg/space/point2.h>
#include <vcg/space/point3.h>
#include <vcg/space/box3.h>

namespace vcg
{

int GreatestCommonDivisor(const int a, const int b)
{
int m = a;
int n = b;

do
{
if (m<n) std::swap(m, n);
m = m % n;
std::swap(m, n);
}
while (n!=0);
return m;
}





template < class OBJECT_TYPE, class SCALAR_TYPE >
class PerfectSpatialHashing : public vcg::SpatialIndex< OBJECT_TYPE, SCALAR_TYPE >
{
template < typename TYPE >
struct Dereferencer
{
static				TYPE& Reference(TYPE				 &t)	{	return  t;	}
static				TYPE& Reference(TYPE*			 &t)	{ return *t;  }
static const  TYPE& Reference(const TYPE	 &t)	{ return  t;	}
static const	TYPE& Reference(const TYPE* &t)	{ return *t;	}
};

template < typename TYPE >
struct ReferenceType { typedef TYPE Type; };

template < typename TYPE >
struct ReferenceType< TYPE * > { typedef typename ReferenceType<TYPE>::Type Type; };

public:
typedef						SCALAR_TYPE													ScalarType;
typedef						OBJECT_TYPE													ObjectType;
typedef typename	ReferenceType< ObjectType >::Type * ObjectPointer;
typedef typename  vcg::Box3< ScalarType >							BoundingBoxType;
typedef typename	vcg::Point3< ScalarType >						CoordinateType;

protected:

struct NeighboringEntryIterator
{

NeighboringEntryIterator(const vcg::Point3i &entry, const int table_size)
{
m_Center		= entry;
m_TableSize = table_size;
m_CurrentNeighbor.X() = (m_Center.X()+m_TableSize-1)%m_TableSize;
m_CurrentNeighbor.Y() = m_Center.Y();
m_CurrentNeighbor.Z() = m_Center.Z();
m_CurrentIteration		= 0;
}


void operator++(int)
{
switch(++m_CurrentIteration)
{
case 1: m_CurrentNeighbor.X()=(m_Center.X()+1)%m_TableSize; break;
case 2: m_CurrentNeighbor.X()=m_Center.X(); m_CurrentNeighbor.Y()=(m_Center.Y()+m_TableSize-1)%m_TableSize; break;
case 3: m_CurrentNeighbor.Y()=(m_Center.Y()+1)%m_TableSize; break;
case 4: m_CurrentNeighbor.Y()=m_Center.Y(); m_CurrentNeighbor.Z()=(m_Center.Z()+m_TableSize-1)%m_TableSize; break;
case 5: m_CurrentNeighbor.Z()=(m_Center.Z()+1)%m_TableSize; break;
default: m_CurrentNeighbor = vcg::Point3i(-1, -1, -1); break;
}
}


vcg::Point3i operator*() { return m_CurrentNeighbor; }


NeighboringEntryIterator& operator =(const NeighboringEntryIterator &it)
{
m_Center						= it.m_Center						;
m_CurrentNeighbor		= it.m_CurrentNeighbor	;
m_CurrentIteration	= it.m_CurrentIteration	;
m_TableSize					= it.m_TableSize				;
return *this;
}


inline bool operator <(const int value) { return m_CurrentIteration<value; }

protected:
vcg::Point3i	m_Center;						
vcg::Point3i	m_CurrentNeighbor;	
int						m_CurrentIteration; 
int						m_TableSize;				
}; 






class UniformGrid
{
public:
typedef vcg::Point3i CellCoordinate;


struct EntryIterator
{
friend class UniformGrid;


EntryIterator(UniformGrid *uniform_grid, const CellCoordinate &position)
{
m_UniformGrid			= uniform_grid;
m_CurrentPosition = position;
}



void operator++(int)
{
if (++m_CurrentPosition.Z()==m_UniformGrid->GetResolution())
{
m_CurrentPosition.Z() = 0;
if (++m_CurrentPosition.Y()==m_UniformGrid->GetResolution())
{
m_CurrentPosition.Y() = 0;
if (++m_CurrentPosition.X()==m_UniformGrid->GetResolution())
m_CurrentPosition = CellCoordinate(-1, -1, -1);
}
}
}



void operator =(const EntryIterator &it)
{
m_UniformGrid			= it.m_UniformGrid;
m_CurrentPosition = it.m_CurrentPosition;
}


bool operator==(const EntryIterator &it) const
{
return m_CurrentPosition==it.m_CurrentPosition;
}


bool operator!=(const EntryIterator &it) const
{
return m_CurrentPosition!=it.m_CurrentPosition;
}


std::vector< ObjectPointer >* operator*()
{
return m_UniformGrid->GetObjects(m_CurrentPosition);
}


CellCoordinate GetPosition() const
{
return m_CurrentPosition;
}


protected:
UniformGrid			* m_UniformGrid;
CellCoordinate		m_CurrentPosition;
}; 



UniformGrid() {}


~UniformGrid() {}



EntryIterator Begin() { return EntryIterator(this, CellCoordinate( 0,  0,  0)); }
EntryIterator End()		{ return EntryIterator(this, CellCoordinate(-1, -1, -1)); }



NeighboringEntryIterator GetNeighboringEntryIterator(const CellCoordinate &at) { return NeighboringEntryIterator(at, m_CellPerSide); }



void Allocate(const BoundingBoxType &bounding_box, const int cell_per_side)
{
m_CellPerSide = cell_per_side;
m_BoundingBox = bounding_box;
m_CellSize		= (m_BoundingBox.max - m_BoundingBox.min)/ScalarType(cell_per_side);

m_Grid.resize(m_CellPerSide);
for (int i=0; i<m_CellPerSide; i++)
{
m_Grid[i].resize(m_CellPerSide);
for (int j=0; j<m_CellPerSide; j++)
m_Grid[i][j].resize(m_CellPerSide);
}
}



void Finalize()
{
m_Grid.clear();
}



template < class OBJECT_ITERATOR >
void InsertElements(const OBJECT_ITERATOR &begin, const OBJECT_ITERATOR &end)
{
typedef OBJECT_ITERATOR ObjectIterator;
typedef Dereferencer< typename ReferenceType< typename OBJECT_ITERATOR::value_type >::Type > ObjectDereferencer;

std::vector< CellCoordinate > cells_occupied;
for (ObjectIterator iObject=begin; iObject!=end; iObject++)
{
ObjectPointer pObject = &ObjectDereferencer::Reference( *iObject );
GetCellsIndex( pObject, cells_occupied);
for (std::vector< CellCoordinate >::iterator iCell=cells_occupied.begin(), eCell=cells_occupied.end(); iCell!=eCell; iCell++)
GetObjects( *iCell )->push_back( pObject );
cells_occupied.clear();
}
}



inline CellCoordinate Interize(const CoordinateType &query) const
{
CellCoordinate result;
result.X() = (int) floorf( (query.X()-m_BoundingBox.min.X())/m_CellSize.X() );
result.Y() = (int) floorf( (query.Y()-m_BoundingBox.min.Y())/m_CellSize.Y() );
result.Z() = (int) floorf( (query.Z()-m_BoundingBox.min.Z())/m_CellSize.Z() );
return result;
}


inline vcg::Box3i Interize(const BoundingBoxType &bounding_box) const
{
vcg::Box3i result;
result.min = Interize(bounding_box.min);
result.max = Interize(bounding_box.max);
return result;
}



void GetCellsIndex(const ObjectPointer pObject, std::vector< CellCoordinate > & cells_occupied)
{
BoundingBoxType object_bb;
(*pObject).GetBBox(object_bb);
CoordinateType corner = object_bb.min;

while (object_bb.IsIn(corner))
{
CellCoordinate cell_index;
cell_index.X() = (int) floorf( (corner.X()-m_BoundingBox.min.X())/m_CellSize.X() );
cell_index.Y() = (int) floorf( (corner.Y()-m_BoundingBox.min.Y())/m_CellSize.Y() );
cell_index.Z() = (int) floorf( (corner.Z()-m_BoundingBox.min.Z())/m_CellSize.Z() );
cells_occupied.push_back( cell_index );

if ((corner.X()+=m_CellSize.X())>object_bb.max.X())
{
corner.X() = object_bb.min.X();
if ( (corner.Z()+=m_CellSize.Z())>object_bb.max.Z() )
{
corner.Z() = object_bb.min.Z();
corner.Y() += m_CellSize.Y();
}
}
}
}



int GetNumberOfNotEmptyCells()
{
int number_of_not_empty_cell = 0;
for (int i=0; i<m_CellPerSide; i++)
for (int j=0; j<m_CellPerSide; j++)
for (int k=0; k<m_CellPerSide; k++)
if (GetObjects(i, j, k)->size()>0)
number_of_not_empty_cell++;
return number_of_not_empty_cell;
}


inline int GetResolution() const { return m_CellPerSide; }



std::vector< ObjectPointer >* GetObjects(const int i, const int j, const int k) { return &m_Grid[i][j][k]; }
std::vector< ObjectPointer >* GetObjects(const CellCoordinate &at)							{ return &m_Grid[at.X()][at.Y()][at.Z()];}
std::vector< ObjectPointer >* operator[](const CellCoordinate &at)							{ return &m_Grid[at.X()][at.Y()][at.Z()];}

protected:
std::vector< std::vector< std::vector< std::vector< ObjectPointer > > > >
m_Grid;					
BoundingBoxType	m_BoundingBox;	
int							m_CellPerSide;	
CoordinateType 	m_CellSize;			
}; 







class HashTable
{
public:
typedef vcg::Point3i EntryCoordinate;

struct Data
{

Data(std::vector< ObjectPointer > *data)
{
domain_data = data;
}

std::vector< ObjectPointer >	*domain_data;
};


HashTable() {}


~HashTable() { Clear(true); }



NeighboringEntryIterator GetNeighborintEntryIterator(const EntryCoordinate &at) { return NeighboringEntryIterator(at, m_EntryPerSide); }



void Allocate(const int entry_per_side)
{
m_EntryPerSide = entry_per_side;
m_Table.resize(m_EntryPerSide);
for (int i=0; i<m_EntryPerSide; i++)
{
m_Table[i].resize(m_EntryPerSide);
for (int j=0; j<m_EntryPerSide; j++)
m_Table[i][j].resize(m_EntryPerSide, NULL);
}

BuildFreeEntryList();
}



void Finalize()
{
Data *pData;
for (int i=0; i<m_EntryPerSide; i++)
for (int j=0; j<m_EntryPerSide; j++)
for (int k=0; k<m_EntryPerSide; k++)
if ((pData=GetData(i, j, k))!=NULL)
{
std::vector< ObjectPointer >	*domain_data = pData->domain_data;
pData->domain_data = new std::vector< ObjectPointer>( *domain_data );
}


m_FreeEntries.clear();
}



void BuildFreeEntryList()
{
m_FreeEntries.clear();
for (int i=0; i<m_EntryPerSide; i++)
for (int j=0; j<m_EntryPerSide; j++)
for (int k=0; k<m_EntryPerSide; k++)
{
assert(m_Table[i][j][k]==NULL);
m_FreeEntries.push_back(EntryCoordinate(i, j, k));
}
}


void Clear(bool delete_vectors=false)
{
for (int i=0; i<m_EntryPerSide; i++)
for (int j=0; j<m_EntryPerSide; j++)
for (int k=0; k<m_EntryPerSide; k++)
if (m_Table[i][j][k]!=NULL)
{
if (delete_vectors)
delete m_Table[i][j][k]->domain_data;

delete m_Table[i][j][k];
m_Table[i][j][k] = NULL;
}

m_FreeEntries.clear();
}


std::list< EntryCoordinate >* GetFreeEntryList() { return &m_FreeEntries; }


EntryCoordinate DomainToHashTable(const typename UniformGrid::CellCoordinate &p)
{
EntryCoordinate result;
result.X() = p.X()%m_EntryPerSide;
result.Y() = p.Y()%m_EntryPerSide;
result.Z() = p.Z()%m_EntryPerSide;
return result;
}


void SetEntry(const EntryCoordinate &at, std::vector< ObjectPointer > *data)
{
assert(IsFree(at));
m_Table[at.X()][at.Y()][at.Z()] = new Data(data);
m_FreeEntries.remove(at);
}


void ValidateEntry(EntryCoordinate &entry)
{
while (entry.X()<0) entry.X()+=m_EntryPerSide;
while (entry.Y()<0) entry.Y()+=m_EntryPerSide;
while (entry.Z()<0) entry.Z()+=m_EntryPerSide;
}


inline bool IsFree(const EntryCoordinate &at) const
{
return (GetData(at)==NULL);
}


inline int GetSize() { return m_EntryPerSide; }


inline int GetNumberOfFreeEntries()
{
return int(m_FreeEntries.size());
}


inline int GetNumberOfNotEmptyEntries()
{
return (int(powf(float(m_EntryPerSide), 3.0f))-int(m_FreeEntries.size()));
}


inline Data* GetData	 (const int i, const int j, const int k) const { return m_Table[i][j][k]; }
inline Data* GetData	 (const EntryCoordinate &at) const { return m_Table[at.X()][at.Y()][at.Z()]; }
inline Data* operator[](const EntryCoordinate &at) const { return m_Table[at.X()][at.Y()][at.Z()]; }

protected:
int																									m_EntryPerSide; 
std::vector< std::vector< std::vector < Data* > > > m_Table;				
std::list< EntryCoordinate >												m_FreeEntries;  
}; 




class OffsetTable
{
public:
typedef unsigned char						OffsetType;
typedef vcg::Point3<OffsetType>	Offset;
typedef Offset								* OffsetPointer;
typedef vcg::Point3i						EntryCoordinate;


struct PreImage
{

PreImage(EntryCoordinate &at, std::vector< typename UniformGrid::CellCoordinate > *preimage)
{
entry_index = at;
pre_image		= preimage;
cardinality = int(pre_image->size());
}


inline bool operator<(const PreImage &second) const { return (cardinality>second.cardinality); }


std::vector< typename UniformGrid::CellCoordinate >
*	pre_image;		
EntryCoordinate			entry_index;  
int									cardinality;	
}; 



OffsetTable() { m_EntryPerSide=-1; m_NumberOfOccupiedEntries=0;}


~OffsetTable() { Clear(); }


void Clear()
{
for (int i=0; i<m_EntryPerSide; i++)
for (int j=0; j<m_EntryPerSide; j++)
for (int k=0; k<m_EntryPerSide; k++)
if (m_Table[i][j][k]!=NULL)
{
delete m_Table[i][j][k];
m_Table[i][j][k] = NULL;
}
m_EntryPerSide = -1;
m_H1PreImage.clear();
m_NumberOfOccupiedEntries = 0;
}


void Allocate(int size)
{
m_NumberOfOccupiedEntries = 0;

m_EntryPerSide = size;
m_Table.resize(m_EntryPerSide);
for (int i=0; i<m_EntryPerSide; i++)
{
m_Table[i].resize(m_EntryPerSide);
for (int j=0; j<m_EntryPerSide; j++)
m_Table[i][j].resize(m_EntryPerSide, NULL);
}

m_H1PreImage.resize(m_EntryPerSide);
for (int i=0; i<m_EntryPerSide; i++)
{
m_H1PreImage[i].resize(m_EntryPerSide);
for (int j=0; j<m_EntryPerSide; j++)
m_H1PreImage[i][j].resize(m_EntryPerSide);
}
}



void Finalize()
{
m_H1PreImage.clear();
}



void BuildH1PreImage(const typename UniformGrid::EntryIterator &begin, const typename UniformGrid::EntryIterator &end)
{
for (typename UniformGrid::EntryIterator iter=begin; iter!=end; iter++)
{
if ((*iter)->size()==0)
continue;

typename UniformGrid::CellCoordinate cell_index = iter.GetPosition();
EntryCoordinate at = DomainToOffsetTable(cell_index);
m_H1PreImage[at.X()][at.Y()][at.Z()].push_back(cell_index);
}
}


void GetPreImageSortedPerCardinality(std::list< PreImage > &pre_image)
{
pre_image.clear();
for (int i=0; i<m_EntryPerSide; i++)
for (int j=0; j<m_EntryPerSide; j++)
for (int k=0; k<m_EntryPerSide; k++)
{
std::vector< typename UniformGrid::CellCoordinate > *preimage = &m_H1PreImage[i][j][k];
if (preimage->size()>0)
pre_image.push_back( PreImage(typename UniformGrid::CellCoordinate(i, j, k), preimage) );
}
pre_image.sort();
}



void SuggestConsistentOffsets(const EntryCoordinate &at, std::vector< Offset > &offsets)
{
offsets.clear();
for (int i=-1; i<2; i++)
for (int j=-1; j<2; j++)
for (int k=-1; k<2; k++)
{
if (i==0 && j==0 && k==0)
continue;

int x = (at.X()+i+m_EntryPerSide)%m_EntryPerSide;
int y = (at.Y()+j+m_EntryPerSide)%m_EntryPerSide;
int z = (at.Z()+k+m_EntryPerSide)%m_EntryPerSide;
EntryCoordinate neighboring_entry(x, y, z);
if (!IsFree(neighboring_entry))
offsets.push_back( *GetOffset(neighboring_entry) );
}
}



void ValidateEntryCoordinate(EntryCoordinate &entry)
{
while (entry.X()<0) entry.X()+=m_EntryPerSide;
while (entry.Y()<0) entry.Y()+=m_EntryPerSide;
while (entry.Z()<0) entry.Z()+=m_EntryPerSide;
}


EntryCoordinate DomainToOffsetTable(const typename UniformGrid::CellCoordinate &coord)
{
EntryCoordinate result;
result.X() = coord.X()%m_EntryPerSide;
result.Y() = coord.Y()%m_EntryPerSide;
result.Z() = coord.Z()%m_EntryPerSide;
return result;
}


void SetOffset(const typename UniformGrid::CellCoordinate &coord, const Offset &offset)
{
EntryCoordinate entry = DomainToOffsetTable(coord);
assert(IsFree(entry));
m_Table[entry.X()][entry.Y()][entry.Z()] = new Offset(offset);
m_NumberOfOccupiedEntries++;
}


void GetRandomOffset( Offset &offset )
{
offset.X() = OffsetType(rand()%m_MAX_VERSOR_LENGTH);
offset.Y() = OffsetType(rand()%m_MAX_VERSOR_LENGTH);
offset.Z() = OffsetType(rand()%m_MAX_VERSOR_LENGTH);
}



inline int	GetSize()			const {return m_EntryPerSide;}



inline bool IsFree(const EntryCoordinate &at) const { return GetOffset(at)==NULL;	}



inline int GetNumberOfOccupiedCells() const { return m_NumberOfOccupiedEntries;		}


inline OffsetPointer& GetOffset (const int i, const int j, const int k)				{ return m_Table[i][j][k]; }
inline OffsetPointer  GetOffset (const int i, const int j, const int k) const { return m_Table[i][j][k]; }

inline OffsetPointer& GetOffset (const EntryCoordinate &at)				{ return m_Table[at.X()][at.Y()][at.Z()]; }
inline OffsetPointer  GetOffset (const EntryCoordinate &at) const { return m_Table[at.X()][at.Y()][at.Z()]; }

inline OffsetPointer& operator[](const EntryCoordinate &at)				{ return m_Table[at.X()][at.Y()][at.Z()]; }
inline OffsetPointer  operator[](const EntryCoordinate &at) const { return m_Table[at.X()][at.Y()][at.Z()]; }

protected:
const static int	m_MAX_VERSOR_LENGTH = 256;	
int								m_EntryPerSide;							
int								m_NumberOfOccupiedEntries;	
std::vector< std::vector< std::vector< OffsetPointer > > >																				m_Table;			
std::vector< std::vector< std::vector< std::vector< typename UniformGrid::CellCoordinate > > > >	m_H1PreImage; 
}; 






class BinaryImage
{
public:

BinaryImage()
{
m_Resolution = -1;
}



~BinaryImage() {}



void Allocate(const int size)
{
m_Resolution = size;
m_Mask.resize(m_Resolution);
for (int i=0; i<m_Resolution; i++)
{
m_Mask[i].resize(m_Resolution);
for (int j=0; j<m_Resolution; j++)
m_Mask[i][j].resize(m_Resolution, false);
}
}



void Clear()
{
for (int i=0; i<m_Resolution; i++)
for (int j=0; j<m_Resolution; j++)
std::fill(m_Mask[i][j].begin(), m_Mask[i][j].end(), false);
}



inline bool ContainsData(const typename UniformGrid::CellCoordinate &at) const { return GetFlag(at)==true;}



inline int GetResolution() const { return m_Resolution; }



inline bool operator()(const int i, const int j, const int k)					{ return m_Mask[i][j][k]; }



inline bool operator[](const typename UniformGrid::CellCoordinate &at)	{ return m_Mask[at.X()][at.Y()][at.Z()]; }
inline const bool& GetFlag(const int i, const int j, const int k)const				{ return m_Mask[i][j][k]; }
inline void  SetFlat(const int i, const int j, const int k)						{ m_Mask[i][j][k] = true; }


inline bool  GetFlag(const typename UniformGrid::CellCoordinate &at)	const			{ return m_Mask[at.X()][at.Y()][at.Z()]; }
inline void  SetFlag(const typename UniformGrid::CellCoordinate &at)						{ m_Mask[at.X()][at.Y()][at.Z()] = true; }

protected:
std::vector< std::vector< std::vector< bool > > >
m_Mask;					
int	m_Resolution;		
}; 






struct Neighbor
{

Neighbor()
{
object		= NULL;
distance = ScalarType(-1.0);
nearest_point.SetZero();
}



Neighbor(ObjectPointer pObject, ScalarType dist, CoordinateType point)
{
object = pObject;
distance = dist;
nearest_point(point);
}



inline bool operator<(const Neighbor &second)
{
return distance<second.distance;
}

ObjectPointer	object;
ScalarType			distance;
CoordinateType nearest_point;
}; 






public:

enum		ConstructionApproach { FastConstructionApproach=0, CompactConstructionApproach=1 };


PerfectSpatialHashing() { srand( (unsigned) time(NULL) ); }


~PerfectSpatialHashing() {  }

template < class OBJECT_ITERATOR >
void Set(const OBJECT_ITERATOR & bObj, const OBJECT_ITERATOR & eObj)
{ Set<OBJECT_ITERATOR>(bObj, eObj, FastConstructionApproach, NULL); }

template < class OBJECT_ITERATOR >
void Set(const OBJECT_ITERATOR & bObj, const OBJECT_ITERATOR & eObj, vcg::CallBackPos *callback)
{ Set<OBJECT_ITERATOR>(bObj, eObj, FastConstructionApproach, callback); }

template < class OBJECT_ITERATOR >
void Set(const OBJECT_ITERATOR & bObj, const OBJECT_ITERATOR & eObj, const ConstructionApproach approach)
{ Set<OBJECT_ITERATOR>(bObj, eObj, approach, NULL); }


template < class OBJECT_ITERATOR >
void Set(const OBJECT_ITERATOR & bObj, const OBJECT_ITERATOR & eObj, const ConstructionApproach approach, vcg::CallBackPos *callback)
{
BoundingBoxType bounding_box;
BoundingBoxType object_bb;
bounding_box.SetNull();
for (OBJECT_ITERATOR iObj=bObj; iObj!=eObj; iObj++)
{
(*iObj).GetBBox(object_bb);
bounding_box.Add(object_bb);
}

BoundingBoxType	resulting_bb(bounding_box);
CoordinateType	offset = bounding_box.Dim()*float(m_BOUNDING_BOX_EXPANSION_FACTOR);
CoordinateType	center = bounding_box.Center();
resulting_bb.Offset(offset);
float longest_side = vcg::math::Max( resulting_bb.DimX(), vcg::math::Max(resulting_bb.DimY(), resulting_bb.DimZ()) )/2.0f;
resulting_bb.Set(center);
resulting_bb.Offset(longest_side);

int number_of_objects = int(std::distance(bObj, eObj));

#ifdef _USE_GRID_UTIL_PARTIONING_
vcg::Point3i resolution;
vcg::BestDim<ScalarType>(number_of_objects, resulting_bb.Dim(), resolution);
int cells_per_side = resolution.X();
#else ifdef _USE_OCTREE_PARTITIONING_ 
int primitives_per_voxel;
int depth = 4;
do
{
int		number_of_voxel = 1<<(3*depth); 
float density					= float(number_of_voxel)/float(depth);
primitives_per_voxel	= int(float(number_of_objects)/density);
depth++;
}
while (primitives_per_voxel>16 && depth<15);
int cells_per_side = int(powf(2.0f, float(depth)));
#endif

m_UniformGrid.Allocate(resulting_bb, cells_per_side);
m_UniformGrid.InsertElements(bObj, eObj);
m_Bitmap.Allocate(cells_per_side);
int number_of_cells_occupied = m_UniformGrid.GetNumberOfNotEmptyCells();

int hash_table_size = (int) ceilf(powf(float(number_of_cells_occupied), 1.0f/float(m_DIMENSION)));
if (hash_table_size>256)
hash_table_size = (int) ceilf(powf(1.01f*float(number_of_cells_occupied), 1.0f/float(m_DIMENSION)));
m_HashTable.Allocate(hash_table_size);

switch (approach)
{
case FastConstructionApproach		:	PerformFastConstruction(number_of_cells_occupied, callback)		; break;
case CompactConstructionApproach: PerformCompactConstruction(number_of_cells_occupied, callback); break;
default: assert(false);
}
Finalize();
} 



template <class OBJECT_POINT_DISTANCE_FUNCTOR, class OBJECT_MARKER, class OBJECT_POINTER_CONTAINER, class DISTANCE_CONTAINER, class POINT_CONTAINER>
unsigned int GetInSphere
(
OBJECT_POINT_DISTANCE_FUNCTOR		&	distance_functor,
OBJECT_MARKER										&	marker,
const CoordinateType									&	sphere_center,
const ScalarType								&	sphere_radius,
OBJECT_POINTER_CONTAINER				&	objects,
DISTANCE_CONTAINER							&	distances,
POINT_CONTAINER									&	points,
bool															sort_per_distance   = true,
bool															allow_zero_distance = true
)
{
BoundingBoxType query_bb(sphere_center, sphere_radius);
vcg::Box3i integer_bb = m_UniformGrid.Interize(query_bb);

vcg::Point3i index;
std::vector< std::vector< ObjectPointer >* > contained_objects;
std::vector< ObjectPointer >* tmp;
for (index.X()=integer_bb.min.X(); index.X()<=integer_bb.max.X(); index.X()++)
for (index.Y()=integer_bb.min.Y(); index.Y()<=integer_bb.max.Y(); index.Y()++)
for (index.Z()=integer_bb.min.Z(); index.Z()<=integer_bb.max.Z(); index.Z()++)
if ((tmp=(*this)[index])!=NULL)
contained_objects.push_back(tmp);

std::vector< Neighbor > results;
for (typename std::vector< typename std::vector< ObjectPointer >* >::iterator iVec=contained_objects.begin(), eVec=contained_objects.end(); iVec!=eVec; iVec++)
for (typename std::vector< ObjectPointer >::iterator iObj=(*iVec)->begin(), eObj=(*iVec)->end(); iObj!=eObj; iObj++ )
{
int r = int(results.size());
results.push_back(Neighbor());
results[r].object		= *iObj;
results[r].distance = sphere_radius;
if (!distance_functor(*results[r].object, sphere_center, results[r].distance, results[r].nearest_point) || (results[r].distance==ScalarType(0.0) && !allow_zero_distance) )
results.pop_back();
}

if (sort_per_distance)
std::sort( results.begin(), results.end() );

int number_of_objects = int(results.size());
distances.resize(number_of_objects);
points.resize(number_of_objects);
objects.resize(number_of_objects);
for (int i=0, size=int(results.size()); i<size; i++)
{
distances[i]	= results[i].distance;
points[i]			= results[i].nearest_point;
objects[i]		= results[i].object;
}
return number_of_objects;
} 



std::vector< ObjectPointer >* operator[](const CoordinateType &query)
{
typename UniformGrid::CellCoordinate ug_index = m_UniformGrid.Interize(query);
if (!m_Bitmap[ug_index])
return NULL;

typename HashTable::EntryCoordinate ht_index = PerfectHashFunction(ug_index);
std::vector< ObjectPointer >* result = m_HashTable[ht_index];
return result;
}


std::vector< ObjectPointer >* operator[](const typename UniformGrid::CellCoordinate  &query)
{
if(!m_Bitmap[query])
return NULL;

typename HashTable::EntryCoordinate ht_index = PerfectHashFunction(query);
std::vector< ObjectPointer >* result = m_HashTable[ht_index]->domain_data;
return result;
}


protected:

typename HashTable::EntryCoordinate PerfectHashFunction(const typename UniformGrid::CellCoordinate &query)
{
typename HashTable::EntryCoordinate result;
typename OffsetTable::OffsetPointer offset = m_OffsetTable[ m_OffsetTable.DomainToOffsetTable(query) ];
result = m_HashTable.DomainToHashTable( Shift(query, *offset) );
return result;
}



typename HashTable::EntryCoordinate Shift(const vcg::Point3i &entry, const typename OffsetTable::Offset &offset)
{
typename HashTable::EntryCoordinate result;
result.X() = entry.X() + int(offset.X());
result.Y() = entry.Y() + int(offset.Y());
result.Z() = entry.Z() + int(offset.Z());
return result;
}



void Finalize()
{
#ifdef _DEBUG
for (UniformGrid::EntryIterator iUGEntry=m_UniformGrid.Begin(), eUGEntry=m_UniformGrid.End(); iUGEntry!=eUGEntry; iUGEntry++)
assert(m_Bitmap.ContainsData(iUGEntry.GetPosition())==((*iUGEntry)->size()>0));
#endif
m_HashTable.Finalize();
m_UniformGrid.Finalize();
m_OffsetTable.Finalize();
}



bool IsAValidOffset(const std::vector< typename UniformGrid::CellCoordinate > *pre_image, const typename OffsetTable::Offset &offset)
{
int ht_size			= m_HashTable.GetSize();
int sqr_ht_size = ht_size*ht_size;
std::vector< int > involved_entries;
for (int i=0, pre_image_size=int((*pre_image).size()); i<pre_image_size; i++)
{
typename UniformGrid::CellCoordinate domain_entry = (*pre_image)[i];
typename HashTable::EntryCoordinate  hash_entry		= m_HashTable.DomainToHashTable( Shift(domain_entry, offset) );
if (!m_HashTable.IsFree(hash_entry))
return false;
else
involved_entries.push_back(hash_entry.X()*sqr_ht_size + hash_entry.Y()*ht_size + hash_entry.Z());
}

std::sort(involved_entries.begin(), involved_entries.end());
for (int i=0, j=1, size=int(involved_entries.size()); j<size; i++, j++)
if (involved_entries[i]==involved_entries[j])
return false;

return true;
}



int GetUnefectiveOffsetTableSize(const int hash_table_size, const int offset_table_size)
{
int result = offset_table_size;
while (GreatestCommonDivisor(hash_table_size, result)!=1 || hash_table_size%result==0)
result += (hash_table_size%2==0) ? (result%2)+1 : 1;  
return result;
}



void PerformFastConstruction(const int number_of_filled_cells, vcg::CallBackPos *callback)
{
int offset_table_size = (int) ceilf(powf(m_SIGMA()*float(number_of_filled_cells), 1.0f/float(m_DIMENSION())));
int hash_table_size   = m_HashTable.GetSize();
int failed_construction_count = 0;
do
{
offset_table_size += failed_construction_count++;
offset_table_size  = GetUnefectiveOffsetTableSize(hash_table_size, offset_table_size);
}
while(!OffsetTableConstructionSucceded(offset_table_size, callback));
}



void PerformCompactConstruction(const int number_of_filled_cells, vcg::CallBackPos *callback)
{
int min_successfully_dimension	= std::numeric_limits<int>::max();
int hash_table_size							= m_HashTable.GetSize();
int half_hash_table_size				= int(float(hash_table_size)/2.0f);

for (int t=0; t<m_MAX_TRIALS_IN_COMPACT_CONSTRUCTION(); t++)
{
int lower_bound = GetUnefectiveOffsetTableSize(hash_table_size, int(double(rand())/double(RAND_MAX)*half_hash_table_size + 1) );
int upper_bound = GetUnefectiveOffsetTableSize(hash_table_size, int(((double) rand() / (double) RAND_MAX) * hash_table_size + half_hash_table_size));

int candidate_offset_table_size;
int last_tried_size = -1;
while (lower_bound<upper_bound)
{
candidate_offset_table_size = GetUnefectiveOffsetTableSize(hash_table_size, int(floorf((lower_bound+upper_bound)/2.0f)));

if (last_tried_size==candidate_offset_table_size)
break;

if ( OffsetTableConstructionSucceded((last_tried_size=candidate_offset_table_size), callback) )
{
upper_bound = candidate_offset_table_size;
min_successfully_dimension = std::min(candidate_offset_table_size, min_successfully_dimension);
}
else
lower_bound = candidate_offset_table_size;

m_HashTable.Clear();
m_HashTable.BuildFreeEntryList();
m_OffsetTable.Clear();
m_Bitmap.Clear();
}
#ifdef _DEBUD
printf("\nPerfectSpatialHashing: minimum offset table size found at the %d-th iteration was %d\n", (t+1), min_successfully_dimension);
#endif
}

while (!OffsetTableConstructionSucceded(min_successfully_dimension, callback))
{
m_HashTable.Clear();
m_HashTable.BuildFreeEntryList();
m_OffsetTable.Clear();
m_Bitmap.Clear();
}
}




bool OffsetTableConstructionSucceded(const int offset_table_size, vcg::CallBackPos *callback)
{
m_OffsetTable.Allocate(offset_table_size); 
m_OffsetTable.BuildH1PreImage(m_UniformGrid.Begin(), m_UniformGrid.End()); 

std::list< typename OffsetTable::PreImage > preimage_slots;
m_OffsetTable.GetPreImageSortedPerCardinality(preimage_slots);

char msg[128];
sprintf(msg, "Building offset table of resolution %d", m_OffsetTable.GetSize());
int	step = int(preimage_slots.size())/100;
int number_of_slots = int(preimage_slots.size());
int perc = 0;
int iter = 0;
for (typename std::list< typename OffsetTable::PreImage >::iterator iPreImage=preimage_slots.begin(), ePreImage=preimage_slots.end(); iPreImage!=ePreImage; iPreImage++, iter++)
{
if (callback!=NULL && iter%step==0 && (perc=iter*100/number_of_slots)<100) (*callback)(perc, msg);

bool found_valid_offset = false;
typename OffsetTable::Offset candidate_offset;

std::vector< typename OffsetTable::Offset > consistent_offsets;
m_OffsetTable.SuggestConsistentOffsets( (*iPreImage).entry_index, consistent_offsets);
for (typename std::vector< typename OffsetTable::Offset >::iterator iOffset=consistent_offsets.begin(), eOffset=consistent_offsets.end(); iOffset!=eOffset && !found_valid_offset; iOffset++)
if (IsAValidOffset(iPreImage->pre_image, *iOffset))
{
found_valid_offset = true;
candidate_offset = *iOffset;
}


if (!found_valid_offset)
{
std::vector< typename UniformGrid::CellCoordinate > *pre_image = (*iPreImage).pre_image;
for (typename std::vector< typename UniformGrid::CellCoordinate >::iterator iPreImage=pre_image->begin(), ePreImage=pre_image->end(); iPreImage!=ePreImage && !found_valid_offset; iPreImage++)
for (NeighboringEntryIterator iUGNeighbourhood=m_UniformGrid.GetNeighboringEntryIterator(*iPreImage); iUGNeighbourhood<6 && !found_valid_offset; iUGNeighbourhood++ )
if (!m_OffsetTable.IsFree( m_OffsetTable.DomainToOffsetTable( *iUGNeighbourhood ) ))
{
typename HashTable::EntryCoordinate ht_entry = PerfectHashFunction(*iUGNeighbourhood);
for (NeighboringEntryIterator iHTNeighbourhood=m_HashTable.GetNeighborintEntryIterator(ht_entry); iHTNeighbourhood<6 && !found_valid_offset; iHTNeighbourhood++)
if (m_HashTable.IsFree(*iHTNeighbourhood))
{
candidate_offset.Import( *iHTNeighbourhood-m_HashTable.DomainToHashTable(*iPreImage) ) ;
if (IsAValidOffset(pre_image, candidate_offset))
found_valid_offset = true;
}
}
}

if (!found_valid_offset)
{
for (int i=0; i<m_MAX_NUM_OF_RANDOM_GENERATED_OFFSET() && !found_valid_offset; i++)
{
typename HashTable::EntryCoordinate base_entry = (*iPreImage).pre_image->at(0);
do
m_OffsetTable.GetRandomOffset(candidate_offset);
while (!m_HashTable.IsFree( m_HashTable.DomainToHashTable( Shift(base_entry, candidate_offset) ) ));

if (IsAValidOffset( (*iPreImage).pre_image, candidate_offset))
found_valid_offset = true;
}

for (typename std::list< typename HashTable::EntryCoordinate >::const_iterator iFreeCell=m_HashTable.GetFreeEntryList()->begin(), eFreeCell=m_HashTable.GetFreeEntryList()->end(); iFreeCell!=eFreeCell && !found_valid_offset; iFreeCell++)
{
typename UniformGrid::CellCoordinate domain_entry		= (*iPreImage).pre_image->at(0);
typename OffsetTable::EntryCoordinate offset_entry		= m_OffsetTable.DomainToOffsetTable(domain_entry);
typename HashTable::EntryCoordinate hashtable_entry	= m_HashTable.DomainToHashTable(domain_entry);
candidate_offset.Import(*iFreeCell - hashtable_entry);

if ( IsAValidOffset(iPreImage->pre_image, candidate_offset) )
found_valid_offset = true;
}
}

if (found_valid_offset)
{
m_OffsetTable.SetOffset( (*iPreImage->pre_image).at(0), candidate_offset);
for (int c=0, pre_image_cardinality = iPreImage->cardinality; c<pre_image_cardinality; c++)
{
typename HashTable::EntryCoordinate ht_entry = PerfectHashFunction( (*iPreImage->pre_image).at(c));
std::vector< ObjectPointer > *domain_data = m_UniformGrid[ (*iPreImage->pre_image).at(c) ];
m_HashTable.SetEntry(ht_entry, domain_data ); 
m_Bitmap.SetFlag((*iPreImage->pre_image).at(c));
}
}
else
{
m_OffsetTable.Clear();
m_HashTable.Clear();
m_HashTable.BuildFreeEntryList();
m_Bitmap.Clear();
return false;
}
}

if (callback!=NULL) (*callback)(100, msg);
return true;
} 





protected:
UniformGrid m_UniformGrid;	
OffsetTable m_OffsetTable;	
HashTable		m_HashTable;		
BinaryImage	m_Bitmap;

static float  m_BOUNDING_BOX_EXPANSION_FACTOR() { return SCALAR_TYPE(0.035); }
static float  m_SIGMA() {return SCALAR_TYPE(1.0f/(2.0f*SCALAR_TYPE(m_DIMENSION)));}
static int    m_MAX_TRIALS_IN_COMPACT_CONSTRUCTION () { return 5; }
static int    m_MAX_NUM_OF_RANDOM_GENERATED_OFFSET() {return 32;}
static int    m_DIMENSION() {return 3;}
}; 


}


#endif 
