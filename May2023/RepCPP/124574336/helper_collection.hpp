#ifndef BOOST_ARCHIVE_DETAIL_HELPER_COLLECTION_HPP
#define BOOST_ARCHIVE_DETAIL_HELPER_COLLECTION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif




#include <cstddef> 
#include <vector>
#include <utility>
#include <memory>
#include <algorithm>

#include <boost/config.hpp>

#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>

namespace boost {

namespace archive {
namespace detail {

class helper_collection
{
helper_collection(const helper_collection&);              
helper_collection& operator = (const helper_collection&); 


typedef std::pair<
const void *,
boost::shared_ptr<void>
> helper_value_type;
template<class T>
boost::shared_ptr<void> make_helper_ptr(){
return boost::make_shared<T>();
}

typedef std::vector<helper_value_type> collection;
collection m_collection;

struct predicate {
BOOST_DELETED_FUNCTION(predicate & operator=(const predicate & rhs))
public:
const void * const m_ti;
bool operator()(helper_value_type const &rhs) const {
return m_ti == rhs.first;
}
predicate(const void * ti) :
m_ti(ti)
{}
};
protected:
helper_collection(){}
~helper_collection(){}
public:
template<typename Helper>
Helper& find_helper(void * const id = 0) {
collection::const_iterator it =
std::find_if(
m_collection.begin(),
m_collection.end(),
predicate(id)
);

void * rval = 0;
if(it == m_collection.end()){
m_collection.push_back(
std::make_pair(id, make_helper_ptr<Helper>())
);
rval = m_collection.back().second.get();
}
else{
rval = it->second.get();
}
return *static_cast<Helper *>(rval);
}
};

} 
} 
} 

#endif 
