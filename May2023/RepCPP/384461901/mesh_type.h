

#pragma once

#include<vcg/simplex/vertex/base.h>						

#include<vcg/simplex/face/base.h>

#include<vcg/complex/complex.h>


#include<wrap/io_trimesh/import_PLY.h>


class DummyEdge;
class StraightFace;





class StraightVertex: public vcg::VertexSimp2< StraightVertex, DummyEdge, StraightFace, vcg::vert::Coord3f,vcg::vert::VFAdj,vcg::vert::Normal3f,vcg::vert::BitFlags>{};


class StraightFace: public vcg::FaceSimp2< StraightVertex, DummyEdge, StraightFace,  vcg::	face::VertexRef,  vcg::	face::FFAdj,  vcg::	face::VFAdj,vcg::	face::Normal3f,vcg::face::BitFlags > {};


class MyStraightMesh: public vcg::tri::TriMesh< std::vector<StraightVertex>,std::vector<StraightFace> >{};


