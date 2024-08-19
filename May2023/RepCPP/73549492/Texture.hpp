#pragma once

#include <vector>

#include "Singleton.hpp"





class Texture : public Singleton<Texture> {

public:

typedef unsigned int Id;


public:

static std::shared_ptr<Texture> Create();

void UploadData();
std::vector<float> &GetData();

~Texture();


private:

void Load();


private:

Id mId = 0;
std::vector<float> mData;

};


