#pragma once

#include <string.h>
#include <vector>
#include <stdio.h>
#include <cstddef>


#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/update/topology.h>
#include <vcg/complex/algorithms/update/bounding.h>
#include <vcg/complex/algorithms/update/quality.h>
#include <vcg/complex/algorithms/update/flag.h>
#include <vcg/complex/algorithms/update/color.h>
#include <vcg/complex/algorithms/update/curvature.h>
#include <vcg/complex/algorithms/update/normal.h>

#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/intersection.h>
#include <vcg/space/index/grid_static_ptr.h>
#include <vcg/space/index/spatial_hashing.h>
#include <vcg/complex/algorithms/closest.h>
#include <vcg/complex/algorithms/smooth.h>
#include <wrap/callback.h>
#include <vcg/complex/append.h>
#include <vcg/container/simple_temporary_data.h>

#include <wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/import_off.h>
#include <wrap/io_trimesh/export.h>
#include <wrap/io_trimesh/export_ply.h>
#include <vcg/complex/algorithms/update/color.h>
#include <vcg/complex/algorithms/update/texture.h>
#include <vcg/complex/algorithms/attribute_seam.h>
#include <vcg/complex/algorithms/refine_loop.h>


using namespace vcg;

class MyFace;
class MyEdge;
class MyVertex;
struct MyUsedTypes: public UsedTypes<Use<MyVertex>::AsVertexType,
Use<MyEdge>::AsEdgeType,
Use<MyFace>::AsFaceType
>{};

class MyEdge : public Edge<MyUsedTypes>{};
class MyVertex  : public Vertex< MyUsedTypes,
vertex::InfoOcf,
vertex::Coord3f,
vertex::BitFlags,
vertex::Normal3f,
vertex::Mark,
vertex::Color4bOcf,
vertex::QualityfOcf,
vertex::VFAdjOcf,
vertex::CurvaturefOcf,
vertex::CurvatureDirfOcf,
vertex::TexCoordfOcf

>{};

class MyFace: public Face  <MyUsedTypes,
face::InfoOcf,
face::VertexRef,
face::BitFlags,
face::Mark,
face::FFAdjOcf,
face::VFAdjOcf,
face::WedgeTexCoordfOcf,
face::Color4bOcf,
face::QualityfOcf,
face::Normal3fOcf

> {};


class MyMesh : public vcg::tri::TriMesh< vcg::vertex::vector_ocf<MyVertex>, vcg::face::vector_ocf<MyFace > >{};
typedef  MyMesh::ScalarType ScalarType;
typedef  MyMesh::VertexIterator VertexIterator;
typedef  MyMesh::VertexPointer VertexPointer;
typedef  MyMesh::FaceIterator FaceIterator;
typedef  MyMesh::FacePointer FacePointer;
typedef  MyMesh::EdgePointer   EdgePointer;
typedef  MyMesh::EdgeIterator   EdgeIterator;
typedef  MyMesh::CoordType CoordType;
typedef  MyMesh::ScalarType ScalarType;
typedef  MyMesh::VertContainer VertContainer;
typedef  MyMesh::FaceContainer FaceContainer;

