
#pragma once



namespace Kratos
{


typedef std::size_t SizeType;





template <SizeType TVoigtSize = 6>
class HighCycleFatigueLawIntegrator
{
public:


KRATOS_CLASS_POINTER_DEFINITION(HighCycleFatigueLawIntegrator);

HighCycleFatigueLawIntegrator()
{
}

HighCycleFatigueLawIntegrator(HighCycleFatigueLawIntegrator const &rOther)
{
}

HighCycleFatigueLawIntegrator &operator=(HighCycleFatigueLawIntegrator const &rOther)
{
return *this;
}

virtual ~HighCycleFatigueLawIntegrator()
{
}




static void CalculateMaximumAndMinimumStresses(
const double CurrentStress,
double& rMaximumStress,
double& rMinimumStress,
const Vector& PreviousStresses,
bool& rMaxIndicator,
bool& rMinIndicator)
{
const double stress_1 = PreviousStresses[1];
const double stress_2 = PreviousStresses[0];
const double stress_increment_1 = stress_1 - stress_2;
const double stress_increment_2 = CurrentStress - stress_1;
if (stress_increment_1 > 1.0e-3 && stress_increment_2 < -1.0e-3) {
rMaximumStress = stress_1;
rMaxIndicator = true;
} else if (stress_increment_1 < -1.0e-3 && stress_increment_2 > 1.0e-3) {
rMinimumStress = stress_1;
rMinIndicator = true;
}
}


static double CalculateTensionCompressionFactor(const Vector& rStressVector)
{
array_1d<double,3> principal_stresses;
AdvancedConstitutiveLawUtilities<6>::CalculatePrincipalStresses(principal_stresses, rStressVector);


double abs_component = 0.0, average_component = 0.0, sum_abs = 0.0, sum_average = 0.0;
for (unsigned int i = 0; i < principal_stresses.size(); ++i) {
abs_component = std::abs(principal_stresses[i]);
average_component = 0.5 * (principal_stresses[i] + abs_component);
sum_average += average_component;
sum_abs += abs_component;
}
const double pre_indicator = sum_average / sum_abs;
if (pre_indicator < 0.5) {
return -1.0;
} else {
return 1.0;
}
}


static double CalculateReversionFactor(const double MaxStress, const double MinStress)
{
return MinStress / MaxStress;
}


static void CalculateFatigueParameters(const double MaxStress,
double ReversionFactor,
const Properties& rMaterialParameters,
double& rB0,
double& rSth,
double& rAlphat,
double& rN_f)
{
const Vector& r_fatigue_coefficients = rMaterialParameters[HIGH_CYCLE_FATIGUE_COEFFICIENTS];
double ultimate_stress = rMaterialParameters.Has(YIELD_STRESS) ? rMaterialParameters[YIELD_STRESS] : rMaterialParameters[YIELD_STRESS_TENSION];
const double yield_stress = ultimate_stress;

const int softening_type = rMaterialParameters[SOFTENING_TYPE];
const int curve_by_points = static_cast<int>(SofteningType::CurveFittingDamage);
if (softening_type == curve_by_points) {
const Vector& stress_damage_curve = rMaterialParameters[STRESS_DAMAGE_CURVE]; 
const SizeType curve_points = stress_damage_curve.size() - 1;

ultimate_stress = 0.0;
for (IndexType i = 1; i <= curve_points; ++i) {
ultimate_stress = std::max(ultimate_stress, stress_damage_curve[i-1]);
}
}

const double Se = r_fatigue_coefficients[0] * ultimate_stress;
const double STHR1 = r_fatigue_coefficients[1];
const double STHR2 = r_fatigue_coefficients[2];
const double ALFAF = r_fatigue_coefficients[3];
const double BETAF = r_fatigue_coefficients[4];
const double AUXR1 = r_fatigue_coefficients[5];
const double AUXR2 = r_fatigue_coefficients[6];

if (std::abs(ReversionFactor) < 1.0) {
rSth = Se + (ultimate_stress - Se) * std::pow((0.5 + 0.5 * ReversionFactor), STHR1);
rAlphat = ALFAF + (0.5 + 0.5 * ReversionFactor) * AUXR1;
} else {
rSth = Se + (ultimate_stress - Se) * std::pow((0.5 + 0.5 / ReversionFactor), STHR2);
rAlphat = ALFAF - (0.5 + 0.5 / ReversionFactor) * AUXR2;
}

const double square_betaf = std::pow(BETAF, 2.0);
if (MaxStress > rSth && MaxStress <= ultimate_stress) {
rN_f = std::pow(10.0,std::pow(-std::log((MaxStress - rSth) / (ultimate_stress - rSth))/rAlphat,(1.0/BETAF)));
rB0 = -(std::log(MaxStress / ultimate_stress) / std::pow((std::log10(rN_f)), square_betaf));

if (softening_type == curve_by_points) {
rN_f = std::pow(rN_f, std::pow(std::log(MaxStress / yield_stress) / std::log(MaxStress / ultimate_stress), 1.0 / square_betaf));
}
}
}


static void CalculateFatigueReductionFactorAndWohlerStress(const Properties& rMaterialParameters,
const double MaxStress,
unsigned int LocalNumberOfCycles,
unsigned int GlobalNumberOfCycles,
const double B0,
const double Sth,
const double Alphat,
double& rFatigueReductionFactor,
double& rWohlerStress)
{
const double BETAF = rMaterialParameters[HIGH_CYCLE_FATIGUE_COEFFICIENTS][4];
if (GlobalNumberOfCycles > 2){
double ultimate_stress = rMaterialParameters.Has(YIELD_STRESS) ? rMaterialParameters[YIELD_STRESS] : rMaterialParameters[YIELD_STRESS_TENSION];

const int softening_type = rMaterialParameters[SOFTENING_TYPE];
const int curve_by_points = static_cast<int>(SofteningType::CurveFittingDamage);
if (softening_type == curve_by_points) {
const Vector& stress_damage_curve = rMaterialParameters[STRESS_DAMAGE_CURVE]; 
const SizeType curve_points = stress_damage_curve.size() - 1;

ultimate_stress = 0.0;
for (IndexType i = 1; i <= curve_points; ++i) {
ultimate_stress = std::max(ultimate_stress, stress_damage_curve[i-1]);
}
}
rWohlerStress = (Sth + (ultimate_stress - Sth) * std::exp(-Alphat * (std::pow(std::log10(static_cast<double>(LocalNumberOfCycles)), BETAF)))) / ultimate_stress;
}
if (MaxStress > Sth) {
rFatigueReductionFactor = std::exp(-B0 * std::pow(std::log10(static_cast<double>(LocalNumberOfCycles)), (BETAF * BETAF)));
rFatigueReductionFactor = (rFatigueReductionFactor < 0.01) ? 0.01 : rFatigueReductionFactor;
}
}






protected:








private:








}; 





} 
