#include "u_delphiClasses.h"
#include "u_Types.h"

#if !defined(KRATOS)
#include <tbb\parallel_for.h>
#include <tbb\blocked_range.h>
using namespace tbb;
#endif

#define ASSIGNMENT_SIZE 2048

typedef void(*TStForLoopElement)(int , int , TObject*);
class TParallelIterator
{
public :

TParallelIterator()
{
}

TParallelIterator(TStForLoopElement ic,  TList<TObject*>* el)  
{  
}
#if !defined(KRATOS)
void operator() ( const blocked_range<size_t>& r ) const 
{ 
for ( size_t i = r.begin(); i != r.end(); ++i ) 
{
TObject* o = elements->elementAt(i);
innercall(i,0 , o);
}

}
#endif

void forloop(int from, int to, TList<TObject*>* elements,TStForLoopElement call)
{

}
};



void assignVertexesAvoidingVisited(TList<TObject*> *vs, TList<TObject*> *vRes ,int iter, int maxAssignment, double minExpQ);
void fastGetSurfaceTriangles(TMesh* aModel);
double ParallelEvaluateClusterByNode(TMesh *aMesh , TVertexesEvaluator fc, double minExpAngle = 500000);
double ParallelEvaluateClusterByEdge(TMesh *aMesh , TVertexesEvaluator fc, double minExpAngle = 500000);
double ParallelEvaluateClusterByFace(TMesh *aMesh , TVertexesEvaluator fc, double minExpAngle = 500000);
double ParallelSmoothMesh(TMesh *aMesh , TVertexesEvaluator fc, double minExpAngle =500000) ;
double ParallelSmoothMeshInGPU(TMesh *aMesh , TVertexesEvaluator fc, double minExpAngle );
void parallelFor(int from, int to,  TList<TObject*>* elements,TStForLoopElement call);
void enterCritical();
void leaveCritical();
void clearPool() ;
void preparePool();
int incrementalID();
void setNumThreads(int nt);
