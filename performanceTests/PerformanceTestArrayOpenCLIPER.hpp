#ifndef PERFORMANCETESTARRAYOPENCLIPER_HPP
#define PERFORMANCETESTARRAYOPENCLIPER_HPP

#include <OpenCLIPER/PerfTestConfResult.hpp>

class PerformanceTestArrayOpenCLIPER : public virtual OpenCLIPER::PerfTestConfResult {
public:
    struct ConfigTraits : OpenCLIPER::PerfTestConfResult::ConfigTraits {
        unsigned int    size = 512;
        unsigned int    dimGridOrGlobalSize = 0;
        unsigned int    dimBlockOrLocalSize = 32;

        //DeviceTraits(DeviceType t=DEVICE_TYPE_ANY,cl::QueueProperties p=cl::QueueProperties::None): type(t),queueProperties(p) {}
        ConfigTraits() {
        	addSupportedShortOption('a', "arraySize", "set array size", false);
        	addSupportedShortOption('b', "blockSize", "set block size", false);
        }
        virtual void configure() override;
    };

    PerformanceTestArrayOpenCLIPER(int argc, char* argv[]);
    ~PerformanceTestArrayOpenCLIPER();

protected:
    PerformanceTestArrayOpenCLIPER();
private:
    virtual void buildSpecificInfo(void* extraInfo) override;
};
#endif // PERFORMANCETESTARRAYOPENCLIPER_HPP
