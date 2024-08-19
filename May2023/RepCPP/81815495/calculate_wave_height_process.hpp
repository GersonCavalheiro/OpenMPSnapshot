
#if !defined(KRATOS_CALCULATE_WAVE_HEIGHT_PROCESS_H_INCLUDED)
#define KRATOS_CALCULATE_WAVE_HEIGHT_PROCESS_H_INCLUDED



#include "includes/model_part.h"
#include "processes/process.h"
#include <fstream>
#include <iostream>

namespace Kratos
{



class CalculateWaveHeightProcess : public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(CalculateWaveHeightProcess);

CalculateWaveHeightProcess(ModelPart &rModelPart,
const int HeightDirection,
const int PlaneDirection,
const double PlaneCoordinates = 0.0,
const double HeightReference = 0.0,
const double Tolerance = 1.0e-2,
const std::string OutputFileName = "WaveHeight",
const double TimeInterval = 0.0) : mrModelPart(rModelPart), mHeightDirection(HeightDirection),
mPlaneDirection(PlaneDirection), mPlaneCoordinates(PlaneCoordinates),
mHeightReference(HeightReference), mTolerance(Tolerance),
mOutputFileName(OutputFileName), mTimeInterval(TimeInterval)
{
std::ofstream my_file;
const std::string file_name = mOutputFileName + ".txt";
my_file.open(file_name, std::ios_base::trunc);
my_file << "    TIME     Wave Height" << std::endl;
my_file.close();
}

virtual ~CalculateWaveHeightProcess() {}


void operator()()
{
Execute();
}


void Execute() override
{
KRATOS_TRY
const double time = mrModelPart.GetProcessInfo()[TIME];
const int step = mrModelPart.GetProcessInfo()[STEP];

if (time - mPreviousPlotTime > mTimeInterval || step == 1) {
const auto it_node_begin = mrModelPart.NodesBegin();
const int num_threads = ParallelUtilities::GetNumThreads();
std::vector<double> max_vector(num_threads, -1.0);

#pragma omp parallel for
for (int i = 0; i < static_cast<int>(mrModelPart.Nodes().size()); i++) {
auto it_node = it_node_begin + i;

const int thread_id = OpenMPUtils::ThisThread();
const auto &r_node_coordinates = it_node->Coordinates();
if (it_node->IsNot(ISOLATED) &&
it_node->Is(FREE_SURFACE) &&
r_node_coordinates(mPlaneDirection) < mPlaneCoordinates + mTolerance &&
r_node_coordinates(mPlaneDirection) > mPlaneCoordinates - mTolerance)
{
const double height = r_node_coordinates(mHeightDirection);

const double wave_height = std::abs(height - mHeightReference);
if (wave_height > max_vector[thread_id])
max_vector[thread_id] = wave_height;
}
}
const double max_height = *std::max_element(max_vector.begin(), max_vector.end());

if (max_height > -1.0) {
std::ofstream my_file;
const std::string file_name = mOutputFileName + ".txt";
my_file.open(file_name, std::ios_base::app);
my_file << "  " + std::to_string(time) + "    " + std::to_string(max_height - mHeightReference) << std::endl;
mPreviousPlotTime = time;
}

}
KRATOS_CATCH("");
}




std::string Info() const override
{
return "CalculateWaveHeightProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "CalculateWaveHeightProcess";
}


protected:

CalculateWaveHeightProcess(CalculateWaveHeightProcess const &rOther);


private:

ModelPart &mrModelPart;
int mHeightDirection;
int mPlaneDirection;

double mPlaneCoordinates;
double mHeightReference;
double mTolerance;
std::string mOutputFileName;
double mTimeInterval;
double mPreviousPlotTime = 0.0;



CalculateWaveHeightProcess &operator=(CalculateWaveHeightProcess const &rOther);


}; 




inline std::istream &operator>>(std::istream &rIStream,
CalculateWaveHeightProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const CalculateWaveHeightProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
