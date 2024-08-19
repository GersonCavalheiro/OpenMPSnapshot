#ifndef BOOST_LOCALE_DATE_TIME_FACET_HPP_INCLUDED
#define BOOST_LOCALE_DATE_TIME_FACET_HPP_INCLUDED

#include <boost/locale/config.hpp>
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4275 4251 4231 4660)
#endif

#include <boost/cstdint.hpp>
#include <locale>

namespace boost {
namespace locale {
namespace period {
namespace marks {
enum period_mark {
invalid,                    
era,                        
year,                       
extended_year,              
month,                      
day,                        
day_of_year,                
day_of_week,                
day_of_week_in_month,       
day_of_week_local,          
hour,                       
hour_12,                    
am_pm,                      
minute,                     
second,                     
week_of_year,               
week_of_month,              
first_day_of_week,          
};

} 

class period_type {
public:
period_type(marks::period_mark m = marks::invalid) : mark_(m) 
{
}

marks::period_mark mark() const
{
return mark_;
}

bool operator==(period_type const &other) const
{
return mark()==other.mark();
}
bool operator!=(period_type const &other) const
{
return mark()!=other.mark();
}
private:
marks::period_mark mark_;
};

} 

struct posix_time {
int64_t seconds; 
uint32_t nanoseconds;  
};


class abstract_calendar {
public:

typedef enum {
absolute_minimum,   
actual_minimum,     
greatest_minimum,   
current,            
least_maximum,      
actual_maximum,     
absolute_maximum,   
} value_type;

typedef enum {
move,   
roll,   
} update_type;

typedef enum {
is_gregorian,   
is_dst          
} calendar_option_type;

virtual abstract_calendar *clone() const = 0;

virtual void set_value(period::marks::period_mark p,int value) = 0;

virtual void normalize() = 0;

virtual int get_value(period::marks::period_mark p,value_type v) const = 0;

virtual void set_time(posix_time const &p)  = 0;
virtual posix_time get_time() const  = 0;

virtual void set_option(calendar_option_type opt,int v) = 0;
virtual int get_option(calendar_option_type opt) const = 0;

virtual void adjust_value(period::marks::period_mark p,update_type u,int difference) = 0;

virtual int difference(abstract_calendar const *other,period::marks::period_mark p) const = 0;

virtual void set_timezone(std::string const &tz) = 0;
virtual std::string get_timezone() const = 0;

virtual bool same(abstract_calendar const *other) const = 0;

virtual ~abstract_calendar()
{
}

};

class BOOST_LOCALE_DECL calendar_facet : public std::locale::facet {
public:
calendar_facet(size_t refs = 0) : std::locale::facet(refs) 
{
}
virtual abstract_calendar *create_calendar() const = 0;

static std::locale::id id;
};

} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif


#endif

