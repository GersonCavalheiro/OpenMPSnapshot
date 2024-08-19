#include "MainTable.h"
#include "SamplingModel.h"
#include "GraphGenerator.h"
#include "LabelsReshuffler.h"
#include <boost/timer/timer.hpp>
#include <omp.h>


int main(int argc, const char** argv) 
{
omp_set_nested(1);
printf("Running Tool for Analysis of COmponent Systems\n");
#pragma omp parallel
if(omp_get_thread_num()==0) printf("threads: %d\n", omp_get_num_threads());

auto stopwatch = boost::timer::auto_cpu_timer();

MainTable* model;
GraphGenerator* G;

if(argc < 2){
cerr<<"Please write some options"<<endl;
cout<<"0 ---> read mainTable.csv"<<endl;
cout<<"1 ---> read and extimate correlation mainTable.csv"<<endl;
cout<<"2 ---> extimate means and variances"<<endl;
cout<<"3 ---> Do the sampling"<<endl;
cout<<"4 ---> read nullTable.csv"<<endl;
cout<<"5 ---> nullTable.csv read and extimate correlation"<<endl;
cout<<"6 ---> nullTable.csv extimate means and variances"<<endl;
cout<<"7 ---> read and make bipartite graph"<<endl;
cout<<"8 ---> shuffle"<<endl;
}else {
switch (std::atoi(argv[1])) {
case 0:
model = new MainTable();
model->read("mainTable.csv", true);
model->~MainTable();
break;
case 1:
model = new MainTable();
model->read("mainTable.csv", true);
model->ExtimateCorrelations();
model->~MainTable();
break;
case 2:
model = new MainTable();
model->SaveMeansVariances("mainTable.csv", true);
model->~MainTable();
break;
case 3:
model = new SamplingModel();
((SamplingModel *) (model))->GenerateNullData(1);
model->~MainTable();
break;
case 4:
model = new MainTable();
model->readNull("nullTable.csv", true, true);
model->~MainTable();
break;
case 5:
model = new MainTable();
model->readNull("nullTable.csv", true);
model->ExtimateCorrelations("correlations_null.dat");
model->~MainTable();
case 6:
model = new MainTable();
model->readNull("nullTable.csv", false, true);
model->~MainTable();
break;
case 7:
G = new GraphGenerator(20000, 1.1 ,true, true);
G->MakeGraph();
delete G;
break;
case 8:
LabelsReshuffler::Shuffle();
break;
default:
std::cerr << "missing arguments" << std::endl;
break;
}

}

return 0;
}
