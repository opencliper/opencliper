#ifndef PERFORMANCETESTARRAYOPPARALLEL_HPP
#define PERFORMANCETESTARRAYOPPARALLEL_HPP

#include <LPISupport/PerfTestConfResult.hpp>

class PerformanceTestArrayOpParallel : public LPISupport::PerfTestConfResult {
public:
    struct ConfigTraits : LPISupport::PerfTestConfResult::ConfigTraits {
        unsigned int    size = 512;
        unsigned int    dimGridOrGlobalSize = 0;
        unsigned int    dimBlockOrLocalSize = 32;

        //DeviceTraits(DeviceType t=DEVICE_TYPE_ANY,cl::QueueProperties p=cl::QueueProperties::None): type(t),queueProperties(p) {}
        ConfigTraits() {
        	addSupportedShortOption('a', "arraySize", "set array size", false);
        	addSupportedShortOption('b', "blockSize", "set block size", false);
        }
        virtual void configure();
    };

    PerformanceTestArrayOpParallel(int argc, char* argv[]);
    ~PerformanceTestArrayOpParallel();

protected:
    PerformanceTestArrayOpParallel();
private:
    virtual void buildSpecificInfo(void* extraInfo);
};

#endif // PERFORMANCETESTARRAYOPPARALLEL_HPP
