#ifndef APPLYMASK_HPP
#define APPLYMASK_HPP

#include <OpenCLIPER/Process.hpp>
#include <OpenCLIPER/KData.hpp>

namespace OpenCLIPER {

class ApplyMask: public Process {
    public:
	struct LaunchParameters: Process::LaunchParameters {
	    //DataHandle samplingMasksDataHandle=INVALIDDATAHANDLE;
	    std::shared_ptr<SamplingMasksData> samplingMasksData;
	    //DataParametersTypes_t dataParametersTypes;
	    //Parameters(ConjugateSensMap_t c, DataParametersTypes_t t):conjugateSensMap(c), dataParametersTypes(t) {}
	    explicit LaunchParameters(const std::shared_ptr<SamplingMasksData>& m): samplingMasksData(m) {}
	};

	void init();
	void launch();

        const std::string getKernelFile() const { return "applyMask.cl"; }

    private:
	using Process::Process;
};

} // namespace OpenCLIPER
#endif // APPLYMASK_HPP
