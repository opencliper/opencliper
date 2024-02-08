#include <LPISupport/SampleCollection.hpp>
#include <LPISupport/Utils.hpp>
/**
 * @brief ...
 *
 */
namespace LPISupport {

/**
 * @brief Constructor for class (sets the name field and creates a default configuration object)
 *
 * @param[in] name name for the sample collection
 */
SampleCollection::SampleCollection(const std::string &name) {
    setSampleName(name);
    pOutputConfigTraits = std::make_shared<OutputConfigTraits>();
}

/**
 * @brief Constructor for class (sets the name field and stores the pointer to the configuration object parameter)
 *
 * @param[in] name name for the sample collection
 * @param[in] pOutputConfigTraits pointer to object with summary output configuration
 */
SampleCollection::SampleCollection(const std::string &name, std::shared_ptr<OutputConfigTraits> pOutputConfigTraits) {
    setSampleName(name);
    this->pOutputConfigTraits = pOutputConfigTraits;
}

/**
 * @brief Destructor for class
 *
 */
SampleCollection::~SampleCollection() {}

/**
 * @brief Adds two sample collections sample by sample
 *
 * @param[in] newSamples new sample collection to be added to the samples stored in this object
 */
void SampleCollection::addSamples(SampleCollection newSamples) {
    if(newSamples.getNumOfSamples() != this->getNumOfSamples()) {
	BTTHROW(std::invalid_argument("Number of samples in collection to be added is different from number of samples in current collection"),
		"SampleCollection::addSamples");
    }
    for(unsigned int i = 0; i < this->getNumOfSamples(); i++) {
	samples.at(i) += newSamples.getSample(i);
    }
}

/**
 * @brief Calculates mean of samples
 */
void SampleCollection::calcMean() {
    unsigned int numOfSamples = samples.size();
    double acum = 0.0;
    for(unsigned int i = 0; i < numOfSamples; i++) {
	acum += samples.at(i);
    }
    mean = acum / numOfSamples;
    meanValid = true;
}

/**
 * @brief Calculates variance of samples
 */
void SampleCollection::calcVariance() {
    if(meanValid == false) {
	calcMean();
    }
    unsigned int numOfSamples = samples.size();
    for(unsigned int i = 0; i < numOfSamples; i++) {
	variance += pow((samples.at(i) - mean), 2) ;
    }
    if(numOfSamples == 1) {
	variance = 0;
    }
    else {
	variance = variance / (numOfSamples - 1);
    }
    varianceValid = true;
}

/**
 * @brief Returns the mean of the samples (if stored mean value is not valid, it is recalculated)
 *
 * @return the mean of the samples
 */
double SampleCollection::getMean() {
    if(meanValid == false) {
	calcMean();
    }
    return mean;
}

/**
 * @brief Returns the variance of the samples (if stored variance value is not valid, it is recalculated)
 *
 * @return the variance of the samples
 */
double SampleCollection::getVariance() {
    if(varianceValid == false) {
	calcVariance();
    }
    return variance;
}

/**
 * @brief Builds summary output of samples collection and returns it as an InfoItems object
 *
 * Summary output information depends on pOutputConfigTraits class variable, that stores configuration for summary output.
 * @param[in] numDigitsPrec number of precision digits of double values (that are stored as strings)
 * @return smart pointer to InfoItems object containing sample collection summary output
 */
std::unique_ptr<InfoItems> SampleCollection::to_infoItems(unsigned int numDigitsPrec) {
    std::unique_ptr<InfoItems> pInfoItems = std::unique_ptr<InfoItems>(new InfoItems());
    if(pOutputConfigTraits->showSamples) {
	for(unsigned int i = 0; i < getNumOfSamples(); i ++) {
	    pInfoItems->addInfoItem(sampleName + " #" + std::to_string(i) + " (s)", getSample(i), numDigitsPrec);
	}
    }
    if(pOutputConfigTraits->showMean) {
	pInfoItems->addInfoItem("Mean " + sampleName + " (s)", getMean(), numDigitsPrec);
    }
    if(pOutputConfigTraits->showVariance) {
	pInfoItems->addInfoItem("Variance of " + sampleName, getVariance(), numDigitsPrec);
    }
    return pInfoItems;
}

};
