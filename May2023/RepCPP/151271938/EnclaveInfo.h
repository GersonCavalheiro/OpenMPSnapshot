#pragma once

#include <string>
#include <vector>

#include <sgx_report.h>

namespace sgx {


class EnclaveInfo
{
private:
int enclaveType;
std::string mrEnclaveHex;
std::string mrSignerHex;
std::string productIdHex;
uint32_t securityVersion;
uint64_t attributes;
std::vector<uint8_t> quote;
std::vector<uint8_t> enclaveHeldData;

public:
EnclaveInfo(const sgx_report_t& enclaveReport,
const std::vector<uint8_t>& quoteBuffer,
const std::vector<uint8_t>& enclaveHeldDataIn);

EnclaveInfo(const std::string& jsonPath);

const std::vector<uint8_t>& getQuote() const;

const std::vector<uint8_t>& getEnclaveHeldData() const;

void toJson(const std::string& jsonPath);
};
}
