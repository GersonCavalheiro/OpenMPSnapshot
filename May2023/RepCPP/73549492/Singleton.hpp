#pragma once

#include <memory>





template<typename Type>
class Singleton {

public:

static inline std::shared_ptr<Type> GetSingleton() {
return msSingleton;
}

Singleton(const Singleton<Type> &other) = delete;
void operator=(const Singleton<Type> &other) = delete;


protected:

static void SetSingleton(std::shared_ptr<Type> singleton) {
msSingleton = singleton;
}

Singleton() = default;


private:

static std::shared_ptr<Type> msSingleton;

};


template<typename Type> std::shared_ptr<Type> Singleton<Type>::msSingleton;
