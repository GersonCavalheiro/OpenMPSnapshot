#pragma once

#include <enclave/outside/attestation/EnclaveInfo.h>

#include <string>
#include <vector>

namespace sgx {

EnclaveInfo generateQuote(int enclaveId,
const std::vector<uint8_t>& enclaveHeldData);

void validateQuote(const EnclaveInfo& enclaveInfo,
const std::string& attestationProviderUrl);

void attestEnclave(int enclaveId, std::vector<uint8_t> enclaveHeldData);
}
