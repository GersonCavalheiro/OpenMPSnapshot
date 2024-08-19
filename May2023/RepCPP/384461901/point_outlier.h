
#ifndef VCG_TRI_OUTLIERS__H
#define VCG_TRI_OUTLIERS__H

#include <vcg/space/index/kdtree/kdtree.h>

namespace vcg
{

namespace tri
{

template <class MeshType>
class OutlierRemoval
{
public:

typedef typename MeshType::ScalarType					ScalarType;
typedef typename vcg::KdTree<ScalarType>				KdTreeType;
typedef typename vcg::KdTree<ScalarType>::PriorityQueue	PriorityQueue;



static void ComputeLoOPScore(MeshType& mesh, KdTreeType& kdTree, int kNearest)
{
vcg::tri::RequireCompactness(mesh);
typename MeshType::template PerVertexAttributeHandle<ScalarType> outlierScore = tri::Allocator<MeshType>:: template GetPerVertexAttribute<ScalarType>(mesh, std::string("outlierScore"));
typename MeshType::template PerVertexAttributeHandle<ScalarType> sigma =        tri::Allocator<MeshType>:: template GetPerVertexAttribute<ScalarType>(mesh, std::string("sigma"));
typename MeshType::template PerVertexAttributeHandle<ScalarType> plof =         tri::Allocator<MeshType>:: template GetPerVertexAttribute<ScalarType>(mesh, std::string("plof"));

#pragma omp parallel for schedule(dynamic, 10) 
for (int i = 0; i < (int)mesh.vert.size(); i++)
{
PriorityQueue queue;
kdTree.doQueryK(mesh.vert[i].cP(), kNearest, queue);
ScalarType sum = 0;
for (int j = 0; j < queue.getNofElements(); j++)
sum += queue.getWeight(j);
sum /= (queue.getNofElements());
sigma[i] = sqrt(sum);
}

float mean = 0;
#pragma omp parallel for reduction(+: mean) schedule(dynamic, 10)
for (int i = 0; i < (int)mesh.vert.size(); i++)
{
PriorityQueue queue;
kdTree.doQueryK(mesh.vert[i].cP(), kNearest, queue);
ScalarType sum = 0;
for (int j = 0; j < queue.getNofElements(); j++)
sum += sigma[queue.getIndex(j)];
sum /= (queue.getNofElements());
plof[i] = sigma[i] / sum  - 1.0f;
mean += plof[i] * plof[i];
}

mean /= mesh.vert.size();
mean = sqrt(mean);

#pragma omp parallel for schedule(dynamic, 10)
for (int i = 0; i < (int)mesh.vert.size(); i++)
{
ScalarType value = plof[i] / (mean * sqrt(2.0f));
double dem = 1.0 + 0.278393 * value;
dem += 0.230389 * value * value;
dem += 0.000972 * value * value * value;
dem += 0.078108 * value * value * value * value;
ScalarType op = std::max(0.0, 1.0 - 1.0 / dem);
outlierScore[i] = op;
}

tri::Allocator<MeshType>::DeletePerVertexAttribute(mesh, std::string("sigma"));
tri::Allocator<MeshType>::DeletePerVertexAttribute(mesh, std::string("plof"));
};


static int SelectLoOPOutliers(MeshType& mesh, KdTreeType& kdTree, int kNearest, float threshold)
{
ComputeLoOPScore(mesh, kdTree, kNearest);
int count = 0;
typename MeshType:: template PerVertexAttributeHandle<ScalarType> outlierScore = tri::Allocator<MeshType>::template GetPerVertexAttribute<ScalarType>(mesh, std::string("outlierScore"));
for (int i = 0; i < mesh.vert.size(); i++)
{
if (outlierScore[i] > threshold)
{
mesh.vert[i].SetS();
count++;
}
}
return count;
}




static int DeleteLoOPOutliers(MeshType& m, KdTreeType& kdTree, int kNearest, float threshold)
{
SelectLoOPOutliers(m,kdTree,kNearest,threshold);
int ovn = m.vn;

for(typename MeshType::VertexIterator vi=m.vert.begin();vi!=m.vert.end();++vi)
if((*vi).IsS() ) tri::Allocator<MeshType>::DeleteVertex(m,*vi);
tri::Allocator<MeshType>::CompactVertexVector(m);
tri::Allocator<MeshType>::DeletePerVertexAttribute(m, std::string("outlierScore"));
return m.vn - ovn;
}
};

} 

} 

#endif 
