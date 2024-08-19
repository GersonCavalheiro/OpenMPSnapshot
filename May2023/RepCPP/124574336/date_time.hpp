#ifndef BOOST_LOCALE_DATE_TIME_HPP_INCLUDED
#define BOOST_LOCALE_DATE_TIME_HPP_INCLUDED

#include <boost/locale/config.hpp>
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4275 4251 4231 4660)
#endif

#include <boost/locale/hold_ptr.hpp>
#include <boost/locale/date_time_facet.hpp>
#include <boost/locale/formatting.hpp>
#include <boost/locale/time_zone.hpp>
#include <locale>
#include <vector>
#include <stdexcept>


namespace boost {
namespace locale {


class BOOST_SYMBOL_VISIBLE date_time_error : public std::runtime_error {
public:
date_time_error(std::string const &e) : std::runtime_error(e) {}
};


struct date_time_period 
{
period::period_type type;   
int value;                  
date_time_period operator+() const { return *this; }
date_time_period operator-() const { return date_time_period(type,-value); }

date_time_period(period::period_type f=period::period_type(),int v=1) : type(f), value(v) {}
};

namespace period {
inline period_type invalid(){ return period_type(marks::invalid); }
inline period_type era(){ return period_type(marks::era); }
inline period_type year(){ return period_type(marks::year); }
inline period_type extended_year(){ return period_type(marks::extended_year); }
inline period_type month(){ return period_type(marks::month); }
inline period_type day(){ return period_type(marks::day); }
inline period_type day_of_year(){ return period_type(marks::day_of_year); }
inline period_type day_of_week(){ return period_type(marks::day_of_week); }
inline period_type day_of_week_in_month(){ return period_type(marks::day_of_week_in_month); }
inline period_type day_of_week_local(){ return period_type(marks::day_of_week_local); }
inline period_type hour(){ return period_type(marks::hour); }
inline period_type hour_12(){ return period_type(marks::hour_12); }
inline period_type am_pm(){ return period_type(marks::am_pm); }
inline period_type minute(){ return period_type(marks::minute); }
inline period_type second(){ return period_type(marks::second); }
inline period_type week_of_year(){ return period_type(marks::week_of_year); }
inline period_type week_of_month(){ return period_type(marks::week_of_month); }
inline period_type first_day_of_week(){ return period_type(marks::first_day_of_week); }

inline date_time_period era(int v) { return date_time_period(era(),v); } 
inline date_time_period year(int v) { return date_time_period(year(),v); } 
inline date_time_period extended_year(int v) { return date_time_period(extended_year(),v); } 
inline date_time_period month(int v) { return date_time_period(month(),v); } 
inline date_time_period day(int v) { return date_time_period(day(),v); } 
inline date_time_period day_of_year(int v) { return date_time_period(day_of_year(),v); } 
inline date_time_period day_of_week(int v) { return date_time_period(day_of_week(),v); } 
inline date_time_period day_of_week_in_month(int v) { return date_time_period(day_of_week_in_month(),v); } 
inline date_time_period day_of_week_local(int v) { return date_time_period(day_of_week_local(),v); } 
inline date_time_period hour(int v) { return date_time_period(hour(),v); } 
inline date_time_period hour_12(int v) { return date_time_period(hour_12(),v); } 
inline date_time_period am_pm(int v) { return date_time_period(am_pm(),v); } 
inline date_time_period minute(int v) { return date_time_period(minute(),v); } 
inline date_time_period second(int v) { return date_time_period(second(),v); } 
inline date_time_period week_of_year(int v) { return date_time_period(week_of_year(),v); } 
inline date_time_period week_of_month(int v) { return date_time_period(week_of_month(),v); } 
inline date_time_period first_day_of_week(int v) { return date_time_period(first_day_of_week(),v); } 

inline date_time_period january() { return date_time_period(month(),0); }
inline date_time_period february() { return date_time_period(month(),1); }
inline date_time_period march() { return date_time_period(month(),2); }
inline date_time_period april() { return date_time_period(month(),3); }
inline date_time_period may() { return date_time_period(month(),4); }
inline date_time_period june() { return date_time_period(month(),5); }
inline date_time_period july() { return date_time_period(month(),6); }
inline date_time_period august() { return date_time_period(month(),7); }
inline date_time_period september() { return date_time_period(month(),8); }
inline date_time_period october() { return date_time_period(month(),9); }
inline date_time_period november() { return date_time_period(month(),10); }
inline date_time_period december() { return date_time_period(month(),11); }

inline date_time_period sunday() { return date_time_period(day_of_week(),1); }
inline date_time_period monday() { return date_time_period(day_of_week(),2); }
inline date_time_period tuesday() { return date_time_period(day_of_week(),3); }
inline date_time_period wednesday() { return date_time_period(day_of_week(),4); }
inline date_time_period thursday() { return date_time_period(day_of_week(),5); }
inline date_time_period friday() { return date_time_period(day_of_week(),6); }
inline date_time_period saturday() { return date_time_period(day_of_week(),7); }
inline date_time_period am() { return date_time_period(am_pm(),0); }
inline date_time_period pm() { return date_time_period(am_pm(),1); }

inline date_time_period operator+(period::period_type f) 
{
return date_time_period(f);
}
inline date_time_period operator-(period::period_type f)
{
return date_time_period(f,-1);
}

template<typename T>
date_time_period operator*(period::period_type f,T v)
{
return date_time_period(f,v);
}

template<typename T>
date_time_period operator*(T v,period::period_type f)
{
return date_time_period(f,v);
}
template<typename T>
date_time_period operator*(T v,date_time_period f)
{
return date_time_period(f.type,f.value*v);
}

template<typename T>
date_time_period operator*(date_time_period f,T v)
{
return date_time_period(f.type,f.value*v);
}


} 


class date_time_period_set {
public:

date_time_period_set()
{
}
date_time_period_set(period::period_type f)
{
basic_[0]=date_time_period(f);
}
date_time_period_set(date_time_period const &fl)
{
basic_[0]=fl;
}
void add(date_time_period f)
{
size_t n=size();
if(n < 4)
basic_[n]=f;
else
periods_.push_back(f);
}
size_t size() const
{
if(basic_[0].type == period::period_type())
return 0;
if(basic_[1].type == period::period_type())
return 1;
if(basic_[2].type == period::period_type())
return 2;
if(basic_[3].type == period::period_type())
return 3;
return 4+periods_.size();
}
date_time_period const &operator[](size_t n) const 
{
if(n >= size())
throw std::out_of_range("Invalid index to date_time_period");
if(n < 4)
return basic_[n];
else
return periods_[n-4];
}
private:
date_time_period basic_[4];
std::vector<date_time_period> periods_;
};


inline date_time_period_set operator+(date_time_period_set const &a,date_time_period_set const &b)
{
date_time_period_set s(a);
for(unsigned i=0;i<b.size();i++)
s.add(b[i]);
return s;
}

inline date_time_period_set operator-(date_time_period_set const &a,date_time_period_set const &b)
{
date_time_period_set s(a);
for(unsigned i=0;i<b.size();i++)
s.add(-b[i]);
return s;
}


class BOOST_LOCALE_DECL calendar {
public:

calendar(std::ios_base &ios);
calendar(std::locale const &l,std::string const &zone);
calendar(std::locale const &l);
calendar(std::string const &zone);
calendar();
~calendar();

calendar(calendar const &other);
calendar const &operator=(calendar const &other);

int minimum(period::period_type f) const;
int greatest_minimum(period::period_type f) const;
int maximum(period::period_type f) const;
int least_maximum(period::period_type f) const;

int first_day_of_week() const;

std::locale get_locale() const;
std::string get_time_zone() const;

bool is_gregorian() const;

bool operator==(calendar const &other) const;
bool operator!=(calendar const &other) const;

private:
friend class date_time;
std::locale locale_;
std::string tz_;
hold_ptr<abstract_calendar> impl_;
};


class BOOST_LOCALE_DECL date_time {
public:

date_time();
date_time(date_time const &other);
date_time(date_time const &other,date_time_period_set const &set);
date_time const &operator=(date_time const &other);
~date_time();

date_time(double time);
date_time(double time,calendar const &cal);
date_time(calendar const &cal);

date_time(date_time_period_set const &set);
date_time(date_time_period_set const &set,calendar const &cal);


date_time const &operator=(date_time_period_set const &f);

void set(period::period_type f,int v);
int get(period::period_type f) const;

int operator/(period::period_type f) const
{
return get(f);
}

date_time operator+(period::period_type f) const
{
return *this+date_time_period(f);
}

date_time operator-(period::period_type f) const
{
return *this-date_time_period(f);
}

date_time const &operator+=(period::period_type f)
{
return *this+=date_time_period(f);
}
date_time const &operator-=(period::period_type f)
{
return *this-=date_time_period(f);
}

date_time operator<<(period::period_type f) const
{
return *this<<date_time_period(f);
}

date_time operator>>(period::period_type f) const
{
return *this>>date_time_period(f);
}

date_time const &operator<<=(period::period_type f)
{
return *this<<=date_time_period(f);
}
date_time const &operator>>=(period::period_type f)
{
return *this>>=date_time_period(f);
}

date_time operator+(date_time_period const &v) const;
date_time operator-(date_time_period const &v) const;
date_time const &operator+=(date_time_period const &v);
date_time const &operator-=(date_time_period const &v);

date_time operator<<(date_time_period const &v) const;
date_time operator>>(date_time_period const &v) const ;
date_time const &operator<<=(date_time_period const &v);
date_time const &operator>>=(date_time_period const &v);

date_time operator+(date_time_period_set const &v) const;
date_time operator-(date_time_period_set const &v) const;
date_time const &operator+=(date_time_period_set const &v);
date_time const &operator-=(date_time_period_set const &v);

date_time operator<<(date_time_period_set const &v) const;
date_time operator>>(date_time_period_set const &v) const ;
date_time const &operator<<=(date_time_period_set const &v);
date_time const &operator>>=(date_time_period_set const &v);

double time() const;
void time(double v);

bool operator==(date_time const &other) const;
bool operator!=(date_time const &other) const;
bool operator<(date_time const &other) const;
bool operator>(date_time const &other) const;
bool operator<=(date_time const &other) const;
bool operator>=(date_time const &other) const;

void swap(date_time &other);

int difference(date_time const &other,period::period_type f) const;

int minimum(period::period_type f) const;
int maximum(period::period_type f) const;

bool is_in_daylight_saving_time() const;

private:
hold_ptr<abstract_calendar> impl_;
};

template<typename CharType>
std::basic_ostream<CharType> &operator<<(std::basic_ostream<CharType> &out,date_time const &t)
{
double time_point = t.time();
uint64_t display_flags = ios_info::get(out).display_flags();
if  (
display_flags == flags::date 
|| display_flags == flags::time 
|| display_flags == flags::datetime 
|| display_flags == flags::strftime
) 
{
out << time_point;
}
else {
ios_info::get(out).display_flags(flags::datetime);
out << time_point;
ios_info::get(out).display_flags(display_flags);
}
return out;
}

template<typename CharType>
std::basic_istream<CharType> &operator>>(std::basic_istream<CharType> &in,date_time &t)
{
double v;
uint64_t display_flags = ios_info::get(in).display_flags();
if  (
display_flags == flags::date 
|| display_flags == flags::time 
|| display_flags == flags::datetime 
|| display_flags == flags::strftime
) 
{
in >> v;
}
else {
ios_info::get(in).display_flags(flags::datetime);
in >> v;
ios_info::get(in).display_flags(display_flags);
}
if(!in.fail())
t.time(v);
return in;
}

class date_time_duration {
public:

date_time_duration(date_time const &first,date_time const &second) :
s_(first),
e_(second)
{
}

int get(period::period_type f) const
{
return start().difference(end(),f);
}

int operator / (period::period_type f) const
{
return start().difference(end(),f);
}

date_time const &start() const { return s_; }
date_time const &end() const { return e_; }
private:
date_time const &s_;
date_time const &e_;
};

inline date_time_duration operator-(date_time const &later,date_time const &earlier)
{
return date_time_duration(earlier,later);
}


namespace period {
inline int era(date_time const &dt) { return dt.get(era()); } 
inline int year(date_time const &dt) { return dt.get(year()); } 
inline int extended_year(date_time const &dt) { return dt.get(extended_year()); } 
inline int month(date_time const &dt) { return dt.get(month()); } 
inline int day(date_time const &dt) { return dt.get(day()); } 
inline int day_of_year(date_time const &dt) { return dt.get(day_of_year()); } 
inline int day_of_week(date_time const &dt) { return dt.get(day_of_week()); } 
inline int day_of_week_in_month(date_time const &dt) { return dt.get(day_of_week_in_month()); } 
inline int day_of_week_local(date_time const &dt) { return dt.get(day_of_week_local()); } 
inline int hour(date_time const &dt) { return dt.get(hour()); } 
inline int hour_12(date_time const &dt) { return dt.get(hour_12()); } 
inline int am_pm(date_time const &dt) { return dt.get(am_pm()); } 
inline int minute(date_time const &dt) { return dt.get(minute()); } 
inline int second(date_time const &dt) { return dt.get(second()); } 
inline int week_of_year(date_time const &dt) { return dt.get(week_of_year()); } 
inline int week_of_month(date_time const &dt) { return dt.get(week_of_month()); } 
inline int first_day_of_week(date_time const &dt) { return dt.get(first_day_of_week()); } 

inline int era(date_time_duration const &dt) { return dt.get(era()); } 
inline int year(date_time_duration const &dt) { return dt.get(year()); } 
inline int extended_year(date_time_duration const &dt) { return dt.get(extended_year()); } 
inline int month(date_time_duration const &dt) { return dt.get(month()); } 
inline int day(date_time_duration const &dt) { return dt.get(day()); } 
inline int day_of_year(date_time_duration const &dt) { return dt.get(day_of_year()); } 
inline int day_of_week(date_time_duration const &dt) { return dt.get(day_of_week()); } 
inline int day_of_week_in_month(date_time_duration const &dt) { return dt.get(day_of_week_in_month()); } 
inline int day_of_week_local(date_time_duration const &dt) { return dt.get(day_of_week_local()); } 
inline int hour(date_time_duration const &dt) { return dt.get(hour()); } 
inline int hour_12(date_time_duration const &dt) { return dt.get(hour_12()); } 
inline int am_pm(date_time_duration const &dt) { return dt.get(am_pm()); } 
inline int minute(date_time_duration const &dt) { return dt.get(minute()); } 
inline int second(date_time_duration const &dt) { return dt.get(second()); } 
inline int week_of_year(date_time_duration const &dt) { return dt.get(week_of_year()); } 
inline int week_of_month(date_time_duration const &dt) { return dt.get(week_of_month()); } 
inline int first_day_of_week(date_time_duration const &dt) { return dt.get(first_day_of_week()); } 


}



} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif


#endif

