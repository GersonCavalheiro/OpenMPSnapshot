#include <iostream>
#include <cstdlib>
#include <string>
#include <list>
#include <vector>
#include <fstream>
#include <time.h>
#include <omp.h>
using namespace std;
struct Parameters{
string from;
string to;
unsigned long dep_time_min;
unsigned long dep_time_max;
unsigned long ar_time_min;
unsigned long ar_time_max;
unsigned long max_layover_time;
unsigned long vacation_time_min;
unsigned long vacation_time_max;
list<string> airports_of_interest;
string flights_file;
string alliances_file;
string work_hard_file;
string play_hard_file;
int nb_threads;
};
struct Flight{
string id;
string from;
string to;
unsigned long take_off_time;
unsigned long land_time;
string company;
float cost;
float discout;
};
struct Travel{
vector<Flight> flights;
};
time_t convert_to_timestamp(int day, int month, int year, int hour, int minute, int seconde);
time_t convert_string_to_timestamp(string s);
void print_params(Parameters &parameters);
void print_flight(Flight& flight);
void read_parameters(Parameters& parameters, int argc, char **argv);
void split_string(vector<string>& result, string line, char separator);
void parse_flight(vector<Flight>& flights, string& line);
void parse_flights(vector<Flight>& flights, string filename);
void parse_alliance(vector<string> &alliance, string line);
void parse_alliances(vector<vector<string> > &alliances, string filename);
bool company_are_in_a_common_alliance(const string& c1, const string& c2, vector<vector<string> >& alliances);
bool has_just_traveled_with_company(Flight& flight_before, Flight& current_flight);
bool has_just_traveled_with_alliance(Flight& flight_before, Flight& current_flight, vector<vector<string> >& alliances);
void apply_discount(Travel & travel, vector<vector<string> >&alliances);
float compute_cost(Travel & travel, vector<vector<string> >&alliances);
void print_alliances(vector<vector<string> > &alliances);
void print_flights(vector<Flight>& flights);
bool nerver_traveled_to(Travel travel, string city);
void print_travel(Travel& travel, vector<vector<string> >&alliances);
void compute_path(vector<Flight>& flights, string to, vector<Travel>& travels, unsigned long t_min, unsigned long t_max, Parameters parameters);
Travel find_cheapest(vector<Travel>& travels, vector<vector<string> >&alliances);
void fill_travel(vector<Travel>& travels, vector<Flight>& flights, string starting_point, unsigned long t_min, unsigned long t_max);
void merge_path(vector<Travel>& travel1, vector<Travel>& travel2);
Travel work_hard(vector<Flight>& flights, Parameters& parameters, vector<vector<string> >& alliances);
void output_work_hard(vector<Flight>& flights, Parameters& parameters, vector<vector<string> >& alliances);
time_t timegm(struct tm *tm);
Travel work_hard(vector<Flight>& flights, Parameters& parameters, vector<vector<string> >& alliances){
vector<Travel> travels;
fill_travel(travels, flights, parameters.from, parameters.dep_time_min, parameters.dep_time_max);
double start = omp_get_wtime();
compute_path(flights, parameters.to, travels, parameters.dep_time_min, parameters.dep_time_max, parameters);
double end = omp_get_wtime();
printf("\nFirst Compute Path Time = %f", end - start); 
vector<Travel> travels_back;
fill_travel(travels_back, flights, parameters.to, parameters.ar_time_min, parameters.ar_time_max);
start = omp_get_wtime();
compute_path(flights, parameters.from, travels_back, parameters.ar_time_min, parameters.ar_time_max, parameters);
end = omp_get_wtime();
printf("\nSecond Compute Path Time = %f", end - start); 
merge_path(travels, travels_back);
Travel go =  find_cheapest(travels, alliances);
return go;
}
void apply_discount(Travel & travel, vector<vector<string> >&alliances){
if(travel.flights.size()>0)
travel.flights[0].discout = 1;
if(travel.flights.size()>1){
for(unsigned int i=1; i<travel.flights.size(); i++){
Flight& flight_before = travel.flights[i-1];
Flight& current_flight = travel.flights[i];
if(has_just_traveled_with_company(flight_before, current_flight)){
flight_before.discout = 0.7;
current_flight.discout = 0.7;
}else if(has_just_traveled_with_alliance(flight_before, current_flight, alliances)){
if(flight_before.discout >0.8)
flight_before.discout = 0.8;
current_flight.discout = 0.8;
}else{
current_flight.discout = 1;
}
}
}
}
float compute_cost(Travel & travel, vector<vector<string> >&alliances){
float result = 0;
apply_discount(travel, alliances);
for(unsigned int i=0; i<travel.flights.size(); i++){
result += (travel.flights[i].cost * travel.flights[i].discout);
}
return result;
}
void compute_path(vector<Flight>& flights, string to, vector<Travel>& travels, unsigned long t_min, unsigned long t_max, Parameters parameters){
vector<Travel> final_travels;
int aux = travels.size();
while(travels.size() > 0)
{
Travel travel = travels.back();
Flight current_city = travel.flights.back();
travels.pop_back();
if(current_city.to == to)
{
final_travels.push_back(travel);
}
else
{
#pragma omp parallel for schedule(guided)
for(unsigned int i=0; i<flights.size(); i++)
{
Flight flight = flights[i];
if(
flight.from == current_city.to &&
flight.take_off_time >= t_min &&
flight.land_time <= t_max &&
(flight.take_off_time > current_city.land_time) &&
flight.take_off_time - current_city.land_time <= parameters.max_layover_time &&
nerver_traveled_to(travel, flight.to)
)
{
Travel newTravel = travel;
newTravel.flights.push_back(flight);
if(flight.to == to)
{
final_travels.push_back(newTravel);
}
else
{
travels.push_back(newTravel);
}
}
}
}
aux = travels.size();
}
travels = final_travels;
}
Travel find_cheapest(vector<Travel>& travels, vector<vector<string> >&alliances){
Travel result;
if(travels.size()>0){
result = travels.back();
travels.pop_back();
}else
return result;
while(travels.size()>0){
Travel temp = travels.back();
travels.pop_back();
if(compute_cost(temp, alliances) < compute_cost(result, alliances)){
result = temp;
}
}
return result;
}
void fill_travel(vector<Travel>& travels, vector<Flight>& flights, string starting_point, unsigned long t_min, unsigned long t_max){
for(unsigned int i=0; i< flights.size(); i++){
if(flights[i].from == starting_point &&
flights[i].take_off_time >= t_min &&
flights[i].land_time <= t_max){
Travel t;
t.flights.push_back(flights[i]);
travels.push_back(t);
}
}
}
void merge_path(vector<Travel>& travel1, vector<Travel>& travel2){
vector<Travel> result;
for(unsigned int i=0; i<travel1.size(); i++){
Travel t1 = travel1[i];
for(unsigned j=0; j<travel2.size(); j++){
Travel t2 = travel2[j];
Flight last_flight_t1 = t1.flights.back();
Flight first_flight_t2 = t2.flights[0];
if(last_flight_t1.land_time < first_flight_t2.take_off_time){
Travel new_travel = t1;
new_travel.flights.insert(new_travel.flights.end(), t2.flights.begin(), t2.flights.end());
result.push_back(new_travel);
}
}
}
travel1 = result;
}
time_t convert_to_timestamp(int day, int month, int year, int hour, int minute, int seconde){
tm time;
time.tm_year = year - 1900;
time.tm_mon = month - 1;
time.tm_mday = day;
time.tm_hour = hour;
time.tm_min = minute;
time.tm_sec = seconde;
return timegm(&time);
}
time_t timegm(struct tm *tm){
time_t ret;
char *tz;
ret = mktime(tm);
return ret;
}
time_t convert_string_to_timestamp(string s){
if(s.size() != 14){
cerr<<"The given string is not a valid timestamp"<<endl;
exit(0);
}else{
int day, month, year, hour, minute, seconde;
day = atoi(s.substr(2,2).c_str());
month = atoi(s.substr(0,2).c_str());
year = atoi(s.substr(4,4).c_str());
hour = atoi(s.substr(8,2).c_str());
minute = atoi(s.substr(10,2).c_str());
seconde = atoi(s.substr(12,2).c_str());
return convert_to_timestamp(day, month, year, hour, minute, seconde);
}
}
void print_params(Parameters &parameters){
cout<<"From : "					<<parameters.from					<<endl;
cout<<"To : "					<<parameters.to						<<endl;
cout<<"dep_time_min : "			<<parameters.dep_time_min			<<endl;
cout<<"dep_time_max : "			<<parameters.dep_time_max			<<endl;
cout<<"ar_time_min : "			<<parameters.ar_time_min			<<endl;
cout<<"ar_time_max : "			<<parameters.ar_time_max			<<endl;
cout<<"max_layover_time : "		<<parameters.max_layover_time		<<endl;
cout<<"vacation_time_min : "	<<parameters.vacation_time_min		<<endl;
cout<<"vacation_time_max : "	<<parameters.vacation_time_max		<<endl;
cout<<"flights_file : "			<<parameters.flights_file			<<endl;
cout<<"alliances_file : "		<<parameters.alliances_file			<<endl;
cout<<"work_hard_file : "		<<parameters.work_hard_file			<<endl;
cout<<"play_hard_file : "		<<parameters.play_hard_file			<<endl;
list<string>::iterator it = parameters.airports_of_interest.begin();
for(; it != parameters.airports_of_interest.end(); it++)
cout<<"airports_of_interest : "	<<*it	<<endl;
cout<<"flights : "				<<parameters.flights_file			<<endl;
cout<<"alliances : "			<<parameters.alliances_file			<<endl;
cout<<"nb_threads : "			<<parameters.nb_threads				<<endl;
}
void print_flight(Flight& flight, ofstream& output){
struct tm * take_off_t, *land_t;
take_off_t = gmtime(((const time_t*)&(flight.take_off_time)));
output<<flight.company<<"-";
output<<""<<flight.id<<"-";
output<<flight.from<<" ("<<(take_off_t->tm_mon+1)<<"/"<<take_off_t->tm_mday<<" "<<take_off_t->tm_hour<<"h"<<take_off_t->tm_min<<"min"<<")"<<"/";
land_t = gmtime(((const time_t*)&(flight.land_time)));
output<<flight.to<<" ("<<(land_t->tm_mon+1)<<"/"<<land_t->tm_mday<<" "<<land_t->tm_hour<<"h"<<land_t->tm_min<<"min"<<")-";
output<<flight.cost<<"$"<<"-"<<flight.discout*100<<"%"<<endl;
}
void read_parameters(Parameters& parameters, int argc, char **argv){
for(int i=0; i<argc; i++){
string current_parameter = argv[i];
if(current_parameter == "-from"){
parameters.from = argv[++i];
}else if(current_parameter == "-arrival_time_min"){
parameters.ar_time_min = convert_string_to_timestamp(argv[++i]);
}else if(current_parameter == "-arrival_time_max"){
parameters.ar_time_max = convert_string_to_timestamp(argv[++i]);
}else if(current_parameter == "-to"){
parameters.to = argv[++i];
}else if(current_parameter == "-departure_time_min"){
parameters.dep_time_min = convert_string_to_timestamp(argv[++i]);
}else if(current_parameter == "-departure_time_max"){
parameters.dep_time_max = convert_string_to_timestamp(argv[++i]);
}else if(current_parameter == "-max_layover"){
parameters.max_layover_time = atol(argv[++i]);
}else if(current_parameter == "-vacation_time_min"){
parameters.vacation_time_min = atol(argv[++i]);
}else if(current_parameter == "-vacation_time_max"){
parameters.vacation_time_max = atol(argv[++i]);
}else if(current_parameter == "-vacation_airports"){
while(i+1 < argc && argv[i+1][0] != '-'){
parameters.airports_of_interest.push_back(argv[++i]);
}
}else if(current_parameter == "-flights"){
parameters.flights_file = argv[++i];
}else if(current_parameter == "-alliances"){
parameters.alliances_file = argv[++i];
}else if(current_parameter == "-work_hard_file"){
parameters.work_hard_file = argv[++i];
}else if(current_parameter == "-play_hard_file"){
parameters.play_hard_file = argv[++i];
}else if(current_parameter == "-nb_threads"){
parameters.nb_threads = atoi(argv[++i]);
omp_set_num_threads(parameters.nb_threads);
}
}
}
void split_string(vector<string>& result, string line, char separator){
while(line.find(separator) != string::npos){
size_t pos = line.find(separator);
result.push_back(line.substr(0, pos));
line = line.substr(pos+1);
}
result.push_back(line);
}
void parse_flight(vector<Flight>& flights, string& line){
vector<string> splittedLine;
split_string(splittedLine, line, ';');
if(splittedLine.size() == 7){
Flight flight;
flight.id = splittedLine[0];
flight.from = splittedLine[1];
flight.take_off_time = convert_string_to_timestamp(splittedLine[2].c_str());
flight.to = splittedLine[3];
flight.land_time = convert_string_to_timestamp(splittedLine[4].c_str());
flight.cost = atof(splittedLine[5].c_str());
flight.company = splittedLine[6];
flights.push_back(flight);
}
}
void parse_flights(vector<Flight>& flights, string filename){
string line = "";
ifstream file;
file.open(filename.c_str());
if(!file.is_open()){
cerr<<"Problem while opening the file "<<filename<<endl;
exit(0);
}
while (!file.eof())
{
getline(file, line);
parse_flight(flights, line);
}
}
void parse_alliance(vector<string> &alliance, string line){
vector<string> splittedLine;
split_string(splittedLine, line, ';');
for(unsigned int i=0; i<splittedLine.size(); i++){
alliance.push_back(splittedLine[i]);
}
}
void parse_alliances(vector<vector<string> > &alliances, string filename){
string line = "";
ifstream file;
file.open(filename.c_str());
if(!file.is_open()){
cerr<<"Problem while opening the file "<<filename<<endl;
exit(0);
}
while (!file.eof())
{
vector<string> alliance;
getline(file, line);
parse_alliance(alliance, line);
alliances.push_back(alliance);
}
}
bool company_are_in_a_common_alliance(const string& c1, const string& c2, vector<vector<string> >& alliances){
bool result = false;
for(unsigned int i=0; i<alliances.size(); i++){
bool c1_found = false, c2_found = false;
for(unsigned int j=0; j<alliances[i].size(); j++){
if(alliances[i][j] == c1) c1_found = true;
if(alliances[i][j] == c2) c2_found = true;
}
result |= (c1_found && c2_found);
}
return result;
}
bool has_just_traveled_with_company(Flight& flight_before, Flight& current_flight){
return flight_before.company == current_flight.company;
}
bool has_just_traveled_with_alliance(Flight& flight_before, Flight& current_flight, vector<vector<string> >& alliances){
return company_are_in_a_common_alliance(current_flight.company, flight_before.company, alliances);
}
void print_alliances(vector<vector<string> > &alliances){
for(unsigned int i=0; i<alliances.size(); i++){
cout<<"Alliance "<<i<<" : ";
for(unsigned int j=0; j<alliances[i].size(); j++){
cout<<"**"<<alliances[i][j]<<"**; ";
}
cout<<endl;
}
}
void print_flights(vector<Flight>& flights, ofstream& output){
for(unsigned int i=0; i<flights.size(); i++)
print_flight(flights[i], output);
}
bool nerver_traveled_to(Travel travel, string city){
for(unsigned int i=0; i<travel.flights.size(); i++)
if(travel.flights[i].from == city || travel.flights[i].to == city)
return false;
return true;
}
void print_travel(Travel& travel, vector<vector<string> >&alliances, ofstream& output){
output<<"Price : "<<compute_cost(travel, alliances)<<endl;
print_flights(travel.flights, output);
output<<endl;
}
void output_work_hard(vector<Flight>& flights, Parameters& parameters, vector<vector<string> >& alliances){
ofstream output;
output.open(parameters.work_hard_file.c_str());
Travel travel = work_hard(flights, parameters, alliances);
output<<"“Work Hard” Proposition :"<<endl;
print_travel(travel, alliances, output);
output.close();
}
int main(int argc, char **argv) {
Parameters parameters;
vector<vector<string> > alliances;
read_parameters(parameters, argc, argv);
vector<Flight> flights;
parse_flights(flights, parameters.flights_file);
parse_alliances(alliances, parameters.alliances_file);
double start = omp_get_wtime(); 
output_work_hard(flights, parameters, alliances);
double end = omp_get_wtime();
printf("\nWork hard time = %f\n", end - start);
}
