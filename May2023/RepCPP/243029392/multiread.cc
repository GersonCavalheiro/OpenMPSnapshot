#include <vector>
#include <map>
#include <string>
#include <math.h>

#include "utils/csv_row/include/CSVIterator.h"
#include "utils/include/utility.h"
#include "utils/include/data_types.h"

#include "Loader.h"
#include "Communicator.h"
#include "Process.h"
#include "Printer.h"
#include "Stats.h"

using namespace std;

int main(int argc, char **argv)
{
if (argc != 3)
{
cout << "Usage: " << argv[0] << " <num_omp_threads> <dataset_dimension>" << endl;
exit(-1);
}
int num_omp_threads = stoi(argv[1]);
string dataset_dim = argv[2];

const string dataset_dir_path = "../dataset/";
const string csv_path = dataset_dir_path + "collisions_" + dataset_dim + ".csv";

int *global_lethAccPerWeek;     
AccPair *global_accAndPerc;     
AccPair *global_boroughWeekAcc; 

map<string, int> cfDictionary;
map<string, int> brghDictionary;

int myrank, num_workers;

MPI_Datatype rowType;

MPI_Op accPairSum;

MPI_Init(&argc, &argv);

MPI_Comm_size(MPI_COMM_WORLD, &num_workers);
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

MPI_Type_create_struct(rowNumVariables, rowLength, rowDisplacements, rowTypes, &rowType);
MPI_Type_commit(&rowType);

MPI_Op_create(pairSum, true, &accPairSum);

vector<Row> localRows;
int my_num_rows;
int numYears, numWeeksPerYear;
int numContributingFactors, numBorough;

int *local_lethAccPerWeek;     
AccPair *local_accAndPerc;     
AccPair *local_boroughWeekAcc; 

double overallBegin = 0, overallDuration = 0; 
double loadBegin = 0, loadDuration = 0;       
double scatterBegin = 0, scatterDuration = 0; 
double procBegin = 0, procDuration = 0;       
double writeBegin = 0, writeDuration = 0;     

const string stats_dir_path = "../stats/";
const string stats_path = stats_dir_path + "stats_multiread_" + dataset_dim + "_" + to_string(num_workers) + "p_" + to_string(num_omp_threads) + "t.csv";
Stats stats(stats_path, myrank, MPI_COMM_WORLD, num_workers, num_omp_threads);

overallBegin = MPI_Wtime();

loadBegin = MPI_Wtime();

cout << "[Proc. " + to_string(myrank) + "] Started loading dataset..." << endl;
Loader loader(csv_path, myrank, MPI_COMM_WORLD);
loader.multiReadDataset(localRows, num_workers);
my_num_rows = localRows.size();

cfDictionary = loader.getCFDict();
brghDictionary = loader.getBRGHDict();

loadDuration = MPI_Wtime() - loadBegin;
stats.setLoadTimes(&loadDuration);

MPI_Barrier(MPI_COMM_WORLD); 

scatterBegin = MPI_Wtime();

Communicator comm(num_workers, myrank, MPI_COMM_WORLD);

comm.mergeDictionary(cfDictionary, 0);
comm.mergeDictionary(brghDictionary, 0);

scatterDuration = MPI_Wtime() - scatterBegin;
stats.setScatterTimes(&scatterDuration);

MPI_Barrier(MPI_COMM_WORLD); 

procBegin = MPI_Wtime();


int myNumYears = loader.getNumYears();
MPI_Allreduce(&myNumYears, &numYears, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
numWeeksPerYear = NUM_WEEKS_PER_YEAR;
numContributingFactors = cfDictionary.size();
numBorough = brghDictionary.size();


if (myrank == 0)
{
global_lethAccPerWeek = new int[numYears * numWeeksPerYear]();
global_accAndPerc = new AccPair[numContributingFactors]();
global_boroughWeekAcc = new AccPair[numBorough * numYears * numWeeksPerYear]();
}

local_lethAccPerWeek = new int[numYears * numWeeksPerYear]();
local_accAndPerc = new AccPair[numContributingFactors]();
local_boroughWeekAcc = new AccPair[numBorough * numYears * numWeeksPerYear]();

cout << "[Proc. " + to_string(myrank) + "] Started processing dataset..." << endl;
Process processer(numYears, numWeeksPerYear, &cfDictionary, &brghDictionary, myrank, MPI_COMM_WORLD);

int dynChunk = (int)round(my_num_rows * 0.02); 
omp_set_num_threads(num_omp_threads);
#pragma omp declare reduction(accPairSum:AccPair \
: omp_out += omp_in)

#pragma omp parallel for default(shared) schedule(dynamic, dynChunk) reduction(+                                                                                                                                \
: local_lethAccPerWeek[:numYears * numWeeksPerYear]) reduction(accPairSum                                                        \
: local_accAndPerc[:numContributingFactors], local_boroughWeekAcc \
[:numBorough * numYears * numWeeksPerYear])
for (int i = 0; i < my_num_rows; i++)
{
processer.processLethAccPerWeek(localRows[i], local_lethAccPerWeek);
processer.processNumAccAndPerc(localRows[i], local_accAndPerc);
processer.processBoroughWeekAcc(localRows[i], local_boroughWeekAcc);
}

MPI_Reduce(local_lethAccPerWeek, global_lethAccPerWeek, (numYears * numWeeksPerYear), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
MPI_Reduce(local_accAndPerc, global_accAndPerc, numContributingFactors, MPI_2INT, accPairSum, 0, MPI_COMM_WORLD);
MPI_Reduce(local_boroughWeekAcc, global_boroughWeekAcc, numBorough * numYears * numWeeksPerYear, MPI_2INT, accPairSum, 0, MPI_COMM_WORLD);

procDuration = MPI_Wtime() - procBegin;
stats.setProcTimes(&procDuration);

MPI_Barrier(MPI_COMM_WORLD); 

if (myrank == 0)
{
writeBegin = MPI_Wtime();
const string result_dir_path = "../results/";
const string result_path = result_dir_path + "result_multiread_" + dataset_dim + ".txt";
Printer printer(result_path, numYears, numWeeksPerYear, &cfDictionary, &brghDictionary, myrank, MPI_COMM_WORLD);

if (!printer.openFile())
{
cout << result_path << ": No such file or directory" << endl;
goto exit;
}

printer.writeLethAccPerWeek(global_lethAccPerWeek);
printer.writeNumAccAndPerc(global_accAndPerc);
printer.writeBoroughWeekAcc(global_boroughWeekAcc);

printer.closeFile();
writeDuration = MPI_Wtime() - writeBegin;
}

stats.setWriteTimes(&writeDuration);

overallDuration = MPI_Wtime() - overallBegin;
stats.setOverallTimes(&overallDuration);

if (myrank == 0)
{
if (!stats.openFile())
{
cout << stats_path << ": No such file or directory" << endl;
goto exit;
}

stats.computeAverages();
stats.printStats();
stats.writeStats();

stats.closeFile();
}

exit:
if (myrank == 0)
{
delete[] global_lethAccPerWeek;
delete[] global_accAndPerc;
delete[] global_boroughWeekAcc;
}

delete[] local_lethAccPerWeek;
delete[] local_accAndPerc;
delete[] local_boroughWeekAcc;

MPI_Finalize();
return 0;
}
