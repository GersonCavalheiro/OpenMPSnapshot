#pragma once

#include "Singleton.hpp"





class Slave : public Singleton<Slave> {

public:

static std::shared_ptr<Slave> Create();

void Run();


private:

bool ShouldExit() const;

};


