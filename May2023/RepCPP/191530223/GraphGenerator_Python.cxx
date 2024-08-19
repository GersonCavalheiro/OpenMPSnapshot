
#include <boost/python.hpp>
#include <boost/timer/timer.hpp>
#include <Python.h>

#include <omp.h>

#include "GraphGenerator.h"
#include "MainTable.h"
#include "SamplingModel.h"
#include "LabelsReshuffler.h"

#include "hSBM_analyser.h"

void makegraph(){
boost::timer::auto_cpu_timer stopwatch;
auto G = new GraphGenerator(1000, 1 ,false, true);
G->MakeGraph();
delete G;
}

void sampling(int statisticsRepeat=1){
boost::timer::auto_cpu_timer stopwatch;
auto model = new SamplingModel();
model->GenerateNullData(statisticsRepeat);
model->~SamplingModel();
delete model;
}

BOOST_PYTHON_FUNCTION_OVERLOADS(sampling_overloads, sampling, 0, 1)

void statistics(bool saveOccurrences=true, bool considerZeros=true){
#pragma omp master
cout<<"Running with "<<omp_get_num_threads()<<" threads"<<endl;
boost::timer::auto_cpu_timer stopwatch;
auto model = new MainTable();
model->read("mainTable.csv", saveOccurrences);
model->SaveMeansVariances("mainTable.csv", considerZeros);
model->~MainTable();
delete model;
}

BOOST_PYTHON_FUNCTION_OVERLOADS(statistics_overloads, statistics, 0, 2)

void reshuffle(){
boost::timer::auto_cpu_timer stopwatch;

LabelsReshuffler::Shuffle();
}

void sample_kullback_liebler(){
hSBM::hsbm::sample_kullback_liebler();
}


BOOST_PYTHON_MODULE(tacos)
{
using namespace boost::python;

def("statistics", statistics, statistics_overloads(
(
boost::python::arg("saveAbundancesOccurrences")=true,
boost::python::arg("considerZeros")=true)
)
);
def("sampling", sampling, sampling_overloads(
boost::python::arg("averageOver")=1)
);
def("makegraph", makegraph);

def("shuffleLabels", reshuffle);

def("topickl", sample_kullback_liebler);
}
