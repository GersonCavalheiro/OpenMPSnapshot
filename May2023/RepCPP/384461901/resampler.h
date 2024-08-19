
#ifndef __VCG_MESH_RESAMPLER
#define __VCG_MESH_RESAMPLER

#include <vcg/complex/algorithms/update/normal.h>
#include <vcg/complex/algorithms/update/bounding.h>
#include <vcg/complex/algorithms/update/component_ep.h>
#include <vcg/complex/algorithms/create/marching_cubes.h>
#include <vcg/space/index/kdtree/kdtree_face.h>

namespace vcg {
namespace tri {







template <class OldMeshType,
class NewMeshType,
class DISTFUNCTOR = vcg::face::PointDistanceBaseFunctor<typename OldMeshType::ScalarType > >
class Resampler : public BasicGrid<typename NewMeshType::ScalarType>
{
typedef typename NewMeshType::ScalarType NewScalarType;
typedef typename NewMeshType::BoxType NewBoxType;
typedef typename NewMeshType::CoordType NewCoordType;
typedef typename NewMeshType::VertexType* NewVertexPointer;
typedef typename NewMeshType::VertexIterator NewVertexIterator;
typedef typename OldMeshType::CoordType OldCoordType;
typedef typename OldMeshType::FaceContainer OldFaceCont;
typedef typename OldMeshType::FaceType OldFaceType;
typedef typename OldMeshType::ScalarType OldScalarType;

class Walker : BasicGrid<typename NewMeshType::ScalarType>
{
private:
typedef int VertexIndex;
typedef vcg::KdTreeFace<OldMeshType> GridType;

protected:

int SliceSize;
int	CurrentSlice;
typedef vcg::tri::EmptyTMark<OldMeshType> MarkerFace;
MarkerFace markerFunctor;

VertexIndex *_x_cs; 
VertexIndex	*_y_cs; 
VertexIndex *_z_cs; 
VertexIndex *_x_ns; 
VertexIndex *_z_ns; 


typedef typename  std::pair<bool,float> field_value;
field_value* _v_cs;
field_value* _v_ns;

NewMeshType	*_newM;
OldMeshType	*_oldM;
GridType _g;

public:
NewScalarType max_dim; 
NewScalarType offset;    
bool DiscretizeFlag; 
bool MultiSampleFlag;
bool AbsDistFlag; 
Walker(const Box3<NewScalarType> &_bbox, Point3i _siz )
{
this->bbox= _bbox;
this->siz=_siz;
this->ComputeDimAndVoxel();

SliceSize = (this->siz.X()+1)*(this->siz.Z()+1);
CurrentSlice = 0;
offset=0;
DiscretizeFlag=false;
MultiSampleFlag=false;
AbsDistFlag=false;

_x_cs = new VertexIndex[ SliceSize ];
_y_cs = new VertexIndex[ SliceSize ];
_z_cs = new VertexIndex[ SliceSize ];
_x_ns = new VertexIndex[ SliceSize ];
_z_ns = new VertexIndex[ SliceSize ];

_v_cs= new field_value[(this->siz.X()+1)*(this->siz.Z()+1)];
_v_ns= new field_value[(this->siz.X()+1)*(this->siz.Z()+1)];

};

~Walker()
{}


NewScalarType V(const Point3i &p)
{
return V(p.V(0),p.V(1),p.V(2));
}


std::pair<bool,NewScalarType> VV(int x,int y,int z)
{
assert ((y==CurrentSlice)||(y==(CurrentSlice+1)));


int index=GetSliceIndex(x,z);

if (y==CurrentSlice) return _v_cs[index];
else return _v_ns[index];
}

NewScalarType V(int x,int y,int z)
{
if(DiscretizeFlag) return VV(x,y,z).second+offset<0?-1:1;
return VV(x,y,z).second+offset;
}
field_value DistanceFromMesh(OldCoordType &pp)
{
OldScalarType dist;
const NewScalarType max_dist = max_dim;
OldCoordType testPt;
this->IPfToPf(pp,testPt);

OldCoordType closestPt;
DISTFUNCTOR PDistFunct;
OldFaceType *f = _g.GetClosest(PDistFunct,markerFunctor,testPt,max_dist,dist,closestPt);

if (f==NULL) return field_value(false,0);
if(AbsDistFlag) return field_value(true,dist);
assert(!f->IsD());
bool retIP;

OldCoordType pip(-1,-1,-1);
retIP=InterpolationParameters(*f,(*f).cN(),closestPt, pip);
assert(retIP); 

const NewScalarType InterpolationEpsilon = 0.00001f;
int zeroCnt=0;
if(pip[0]<InterpolationEpsilon) ++zeroCnt;
if(pip[1]<InterpolationEpsilon) ++zeroCnt;
if(pip[2]<InterpolationEpsilon) ++zeroCnt;
assert(zeroCnt<3);

OldCoordType dir=(testPt-closestPt).Normalize();

NewScalarType signBest;

OldCoordType closestNormV, closestNormF;
if(zeroCnt>0) 
{
closestNormV =  (f->V(0)->cN())*pip[0] + (f->V(1)->cN())*pip[1] + (f->V(2)->cN())*pip[2] ;
signBest =  dir.dot(closestNormV) ;
}
else
{
closestNormF =  f->cN() ;
signBest =  dir.dot(closestNormF) ;
}

if(signBest<0) dist=-dist;

return field_value(true,dist);
}

field_value MultiDistanceFromMesh(OldCoordType &pp)
{
float distSum=0;
int positiveCnt=0; 
const int MultiSample=7;
const OldCoordType   delta[7]={OldCoordType(0,0,0),
OldCoordType( 0.2,  -0.01, -0.02),
OldCoordType(-0.2,   0.01,  0.02),
OldCoordType( 0.01,  0.2,   0.01),
OldCoordType( 0.03, -0.2,  -0.03),
OldCoordType(-0.02, -0.03,  0.2 ),
OldCoordType(-0.01,  0.01, -0.2 )};

for(int qq=0;qq<MultiSample;++qq)
{
OldCoordType pp2=pp+delta[qq];
field_value ff= DistanceFromMesh(pp2);
if(ff.first==false) return field_value(false,0);
distSum += fabs(ff.second);
if(ff.second>0) positiveCnt ++;
}
if(positiveCnt<=MultiSample/2) distSum = -distSum;
return field_value(true, distSum/MultiSample);
}

void ComputeSliceValues(int slice,field_value *slice_values)
{
#pragma omp parallel for schedule(dynamic, 10)
for (int i=0; i<=this->siz.X(); i++)
{
for (int k=0; k<=this->siz.Z(); k++)
{
int index=GetSliceIndex(i,k);
OldCoordType pp(i,slice,k);
if(this->MultiSampleFlag) slice_values[index] = MultiDistanceFromMesh(pp);
else	slice_values[index] = DistanceFromMesh(pp);
}
}
}


void ComputeConsensus(int , field_value *slice_values)
{
float max_dist = min(min(this->voxel[0],this->voxel[1]),this->voxel[2]);
int flippedCnt=0;
int flippedTot=0;
int flippedTimes=0;
do
{
flippedCnt=0;
for (int i=0; i<=this->siz.X(); i++)
{
for (int k=0; k<=this->siz.Z(); k++)
{
int goodCnt=0;
int badCnt=0;
int index=GetSliceIndex(i,k);
int index_l,index_r,index_u,index_d;
if(slice_values[index].first)
{
float curVal= slice_values[index].second;
if(i > 0             ) index_l=GetSliceIndex(i-1,k); else index_l = index;
if(i < this->siz.X() ) index_r=GetSliceIndex(i+1,k); else index_r = index;
if(k > 0             ) index_d=GetSliceIndex(i,k-1); else index_d = index;
if(k < this->siz.Z() ) index_u=GetSliceIndex(i,k+1); else index_u = index;

if(slice_values[index_l].first) { goodCnt++; if(fabs(slice_values[index_l].second - curVal) > max_dist) badCnt++; }
if(slice_values[index_r].first) { goodCnt++; if(fabs(slice_values[index_r].second - curVal) > max_dist) badCnt++; }
if(slice_values[index_u].first) { goodCnt++; if(fabs(slice_values[index_u].second - curVal) > max_dist) badCnt++; }
if(slice_values[index_d].first) { goodCnt++; if(fabs(slice_values[index_d].second - curVal) > max_dist) badCnt++; }

if(badCnt >= goodCnt)  {
slice_values[index].second *=-1.0f;
flippedCnt++;
}
}
}
}
flippedTot+=flippedCnt;
flippedTimes++;
}	while(flippedCnt>0);


#ifdef QT_VERSION
if(flippedTot>0)
qDebug("Flipped %i values in %i times",flippedTot,flippedTimes);
#endif
}
template<class EXTRACTOR_TYPE>
void ProcessSlice(EXTRACTOR_TYPE &extractor)
{
for (int i=0; i<this->siz.X(); i++)
{
for (int k=0; k<this->siz.Z(); k++)
{
bool goodCell=true;
Point3i p1(i,CurrentSlice,k);
Point3i p2=p1+Point3i(1,1,1);
for(int ii=0;ii<2;++ii)
for(int jj=0;jj<2;++jj)
for(int kk=0;kk<2;++kk)
goodCell &= VV(p1[0]+ii,p1[1]+jj,p1[2]+kk).first;

if(goodCell) extractor.ProcessCell(p1, p2);
}
}
}


template<class EXTRACTOR_TYPE>
void BuildMesh(OldMeshType &old_mesh,NewMeshType &new_mesh,EXTRACTOR_TYPE &extractor,vcg::CallBackPos *cb)
{
_newM=&new_mesh;
_oldM=&old_mesh;

tri::UpdateNormal<OldMeshType>::PerFaceNormalized(old_mesh);
tri::UpdateNormal<OldMeshType>::PerVertexAngleWeighted(old_mesh);
int _size=(int)old_mesh.fn*100;

_g.Set(_oldM->face.begin(),_oldM->face.end(),_size);
markerFunctor.SetMesh(&old_mesh);

_newM->Clear();

Begin();
extractor.Initialize();
for (int j=0; j<=this->siz.Y(); j++)
{
if (cb) cb((100*j)/this->siz.Y(),"Marching ");
ProcessSlice<EXTRACTOR_TYPE>(extractor);
NextSlice();
}
extractor.Finalize();
for(NewVertexIterator vi=new_mesh.vert.begin();vi!=new_mesh.vert.end();++vi)
if(!(*vi).IsD())
{
this->IPfToPf((*vi).cP(),(*vi).P());
}
}

int GetSliceIndex(int x,int z)
{
VertexIndex index = x+z*(this->siz.X()+1);
return (index);
}

void NextSlice()
{

memset(_x_cs, -1, SliceSize*sizeof(VertexIndex));
memset(_y_cs, -1, SliceSize*sizeof(VertexIndex));
memset(_z_cs, -1, SliceSize*sizeof(VertexIndex));


std::swap(_x_cs, _x_ns);
std::swap(_z_cs, _z_ns);

std::swap(_v_cs, _v_ns);

CurrentSlice ++;

ComputeSliceValues(CurrentSlice + 1,_v_ns);
}

void Begin()
{

CurrentSlice = 0;

memset(_x_cs, -1, SliceSize*sizeof(VertexIndex));
memset(_y_cs, -1, SliceSize*sizeof(VertexIndex));
memset(_z_cs, -1, SliceSize*sizeof(VertexIndex));
memset(_x_ns, -1, SliceSize*sizeof(VertexIndex));
memset(_z_ns, -1, SliceSize*sizeof(VertexIndex));

ComputeSliceValues(CurrentSlice,_v_cs);
ComputeSliceValues(CurrentSlice+1,_v_ns);
}




bool Exist(const vcg::Point3i &p1, const vcg::Point3i &p2, NewVertexPointer &v)
{
int i = p1.X();
int z = p1.Z();
VertexIndex index = i+z*this->siz.X();

int v_ind = 0;
if (p1.X()!=p2.X()) 
{
if (p1.Y()==CurrentSlice)
{
if (_x_cs[index]!=-1)
{
v_ind = _x_cs[index];
v = &_newM->vert[v_ind];
assert(!v->IsD());
return true;
}

}
else
{
if (_x_ns[index]!=-1)
{
v_ind = _x_ns[index];
v = &_newM->vert[v_ind];
assert(!v->IsD());
return true;
}
}
v = NULL;
return false;
}
else if (p1.Y()!=p2.Y()) 
{
if (_y_cs[index]!=-1)
{
v_ind =_y_cs[index];
v = &_newM->vert[v_ind];
assert(!v->IsD());
return true;
}
else
{
v = NULL;
return false;
}

}
else if (p1.Z()!=p2.Z())
{
if (p1.Y()==CurrentSlice)
{
if ( _z_cs[index]!=-1)
{
v_ind = _z_cs[index];
v = &_newM->vert[v_ind];
assert(!v->IsD());
return true;
}

}
else
{
if (_z_ns[index]!=-1)
{
v_ind = _z_ns[index];
v = &_newM->vert[v_ind];
assert(!v->IsD());
return true;
}
}
v = NULL;
return false;
}
assert (0);
return false;
}

NewCoordType Interpolate(const vcg::Point3i &p1, const vcg::Point3i &p2,int dir)
{
NewScalarType f1 = V(p1);
NewScalarType f2 = V(p2);
NewScalarType u =  f1/(f1-f2);
NewCoordType ret(p1.V(0),p1.V(1),p1.V(2));
ret.V(dir) = p1.V(dir)*(1.f-u) + u*p2.V(dir);
return (ret);
}

void GetXIntercept(const vcg::Point3i &p1, const vcg::Point3i &p2, NewVertexPointer &v)
{
assert(p1.X()+1 == p2.X());
assert(p1.Y()   == p2.Y());
assert(p1.Z()   == p2.Z());

int i = p1.X();
int z = p1.Z();
VertexIndex index = i+z*this->siz.X();
VertexIndex pos=-1;
if (p1.Y()==CurrentSlice)
{
if ((pos=_x_cs[index])==-1)
{
_x_cs[index] = (VertexIndex) _newM->vert.size();
pos = _x_cs[index];
Allocator<NewMeshType>::AddVertices( *_newM, 1 );
v = &_newM->vert[pos];
v->P()=Interpolate(p1,p2,0);
return;
}
}
if (p1.Y()==CurrentSlice+1)
{
if ((pos=_x_ns[index])==-1)
{
_x_ns[index] = (VertexIndex) _newM->vert.size();
pos = _x_ns[index];
Allocator<NewMeshType>::AddVertices( *_newM, 1 );
v = &_newM->vert[pos];
v->P()=Interpolate(p1,p2,0);
return;
}
}
assert(pos>=0);
v = &_newM->vert[pos];
}

void GetYIntercept(const vcg::Point3i &p1, const vcg::Point3i &p2, NewVertexPointer &v)
{
assert(p1.X()   == p2.X());
assert(p1.Y()+1 == p2.Y());
assert(p1.Z()   == p2.Z());

int i = p1.X(); 
int z = p1.Z(); 
VertexIndex index = i+z*this->siz.X();
VertexIndex pos=-1;
if ((pos=_y_cs[index])==-1)
{
_y_cs[index] = (VertexIndex) _newM->vert.size();
pos = _y_cs[index];
Allocator<NewMeshType>::AddVertices( *_newM, 1);
v = &_newM->vert[ pos ];
v->P()=Interpolate(p1,p2,1);
}
assert(pos>=0);
v = &_newM->vert[pos];
}

void GetZIntercept(const vcg::Point3i &p1, const vcg::Point3i &p2, NewVertexPointer &v)
{
assert(p1.X()   == p2.X());
assert(p1.Y()   == p2.Y());
assert(p1.Z()+1 == p2.Z());

int i = p1.X(); 
int z = p1.Z(); 
VertexIndex index = i+z*this->siz.X();

VertexIndex pos=-1;
if (p1.Y()==CurrentSlice)
{
if ((pos=_z_cs[index])==-1)
{
_z_cs[index] = (VertexIndex) _newM->vert.size();
pos = _z_cs[index];
Allocator<NewMeshType>::AddVertices( *_newM, 1 );
v = &_newM->vert[pos];
v->P()=Interpolate(p1,p2,2);
return;
}
}
if (p1.Y()==CurrentSlice+1)
{
if ((pos=_z_ns[index])==-1)
{
_z_ns[index] = (VertexIndex) _newM->vert.size();
pos = _z_ns[index];
Allocator<NewMeshType>::AddVertices( *_newM, 1 );
v = &_newM->vert[pos];
v->P()=Interpolate(p1,p2,2);
return;
}
}
assert(pos>=0);
v = &_newM->vert[pos];
}

};

public:

typedef Walker    MyWalker;

typedef vcg::tri::MarchingCubes<NewMeshType, MyWalker> MyMarchingCubes;

static void Resample(OldMeshType &old_mesh, NewMeshType &new_mesh,  NewBoxType volumeBox, vcg::Point3<int> accuracy,float max_dist, float thr=0, bool DiscretizeFlag=false, bool MultiSampleFlag=false, bool AbsDistFlag=false, vcg::CallBackPos *cb=0 )
{
vcg::tri::UpdateBounding<OldMeshType>::Box(old_mesh);

MyWalker	walker(volumeBox,accuracy);

walker.max_dim=max_dist+fabs(thr);
walker.offset = - thr;
walker.DiscretizeFlag = DiscretizeFlag;
walker.MultiSampleFlag = MultiSampleFlag;
walker.AbsDistFlag = AbsDistFlag;
MyMarchingCubes mc(new_mesh, walker);
walker.BuildMesh(old_mesh,new_mesh,mc,cb);
}


};

}
}
#endif
