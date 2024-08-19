#include <iostream>
#include <boost/program_options.hpp>

#include "src/preprocessor.hh"

using namespace std;
namespace po=boost::program_options;

int argParse(int argc,char *argv[],std::tuple<std::string,std::string,std::string,bool,bool,bool>& opts) {
bool timing=false,print=false,serial=false;
std::string output_file_name,cxx_options;
po::options_description description("OMP DAG pragma preprocessor options");
po::positional_options_description positional_opts;
positional_opts.add("input",1);
description.add_options()("help", "print help message")
("input,i", po::value<std::string>(), "input file name")
("output,o", po::value<std::string>(&output_file_name)->default_value(std::string("a.cxx")), "preprocessed file name")
("cxxflags,c", po::value<std::string>(&cxx_options)->default_value(__preprocessor__::Preprocessor::clang_options), "clang cxx flags")
("serial,s", po::bool_switch(&serial), "produce serial code")
("timing,t", po::bool_switch(&timing), "collect and report timing metrics")
("print,p", po::bool_switch(&print), "print the dependency graph");
po::variables_map options;
po::store(po::command_line_parser(argc,argv).options(description).positional(positional_opts).run(), options);
po::notify(options);
if (options.count("help")) {
cout<<description<<endl;
cout<<"************************************************************************************"<<endl;
return 1;
}
std::get<1>(opts)=output_file_name;
std::get<2>(opts)=cxx_options;
std::get<3>(opts)=timing;
std::get<4>(opts)=print;
std::get<5>(opts)=serial;
if (options.count("input"))
std::get<0>(opts)=options["input"].as<std::string>();
else {
cout<<description<<endl;
cout<<"************************************************************************************"<<endl;
return 1;
}
return 0;
}

int main(int argc,char *argv[]) {
using namespace __io__;
std::tuple<std::string,std::string,std::string,bool,bool,bool> opts;
cout<<"************************************************************************************"<<endl;
cout<<"************************************************************************************"<<endl;
cout<<endl<<"                         OMP DAG pragma preprocessor"<<endl<<endl;
if(argParse(argc,argv,opts))
return 1;
cout<<"Options: "<<endl;
cout<<"\tInput file:  "<<std::get<0>(opts)<<endl;
cout<<"\tOutput file: "<<std::get<1>(opts)<<endl;
cout<<"\tCxx flags:   "<<std::get<2>(opts)<<endl;
cout<<"\tSerial code: "<<std::boolalpha<<std::get<5>(opts)<<endl;
cout<<"\tTiming:      "<<std::boolalpha<<std::get<3>(opts)<<endl;
cout<<"\tPrint graph: "<<std::boolalpha<<std::get<4>(opts)<<endl;
cout<<"************************************************************************************"<<endl;
__preprocessor__::Preprocessor preprocesssor;
preprocesssor.parse(std::get<0>(opts),std::get<2>(opts));
preprocesssor.preprocess(std::get<1>(opts),std::get<3>(opts),std::get<4>(opts),std::get<5>(opts));
return 0;
}
