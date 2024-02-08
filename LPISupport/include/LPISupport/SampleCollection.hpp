/* Copyright (C) 2018 Federico Simmross Wattenberg,
 *                    Manuel Rodríguez Cayetano,
 *                    Javier Royuela del Val,
 *                    Elena Martín González,
 *                    Elisa Moya Sáez,
 *                    Marcos Martín Fernández and
 *                    Carlos Alberola López
 *
 * This file is part of OpenCLIPER.
 *
 * OpenCLIPER is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; version 3 of the License.
 *
 * OpenCLIPER is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with OpenCLIPER; If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *  Contact:
 *
 *  Federico Simmross Wattenberg
 *  E.T.S.I. Telecomunicación
 *  Universidad de Valladolid
 *  Paseo de Belén 15
 *  47011 Valladolid, Spain.
 *  fedsim@tel.uva.es
 */
/*
 * Utils.hpp
 *
 *  Created on: 15 de nov. de 2016
 *      Author: manrod
 */

#ifndef INCLUDE_OPENCLIPER_SAMPLECOLLECTION_HPP_
#define INCLUDE_OPENCLIPER_SAMPLECOLLECTION_HPP_
#include <LPISupport/Utils.hpp>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <LPISupport/InfoItems.hpp>

namespace LPISupport {
/**
 * @brief Class that stores a group of samples of some measurement and provides methods for calculation of statistics and storing this information (using an InfoItem object)
 *
 */
class SampleCollection {
    public:
	/// @brief Class for configuring summary output of stored samples
	struct OutputConfigTraits {
	    /// samples are shown in summary output if true
	    bool showSamples = true;
	    /// mean samples is shown in summary output if true
	    bool showMean = true;
	    /// variance of samples is shown in summary output if true
	    bool showVariance = true;
	};

	explicit SampleCollection(const std::string &name);
	SampleCollection(const std::string &name, std::shared_ptr<OutputConfigTraits> pOutputConfigTraits);
	virtual ~SampleCollection();

	/**
	 * @brief Gets the name associated to this sample collection
	 * @return the name of the collection
	 */
	std::string getSampleName() const {
	    return sampleName;
	}
	/**
	 * @brief Sets the name associated to this sample collection
	 * @param[in] sampleName name for the collection
	 */
	void setSampleName(const std::string &sampleName) {
	    this->sampleName = sampleName;
	}
	/**
	 * @brief Removes all the samples of the collection and sets flags showing that mean and variance are not valid
	 */
	void clearSamples() {
	    samples.clear();
	    meanValid = false;
	    varianceValid = false;
	}
	/**
	 * @brief Appends a sample to the collection
	 * @param[in] sample of the sample (double)
	 */
	void appendSample(double sample) {
	    samples.push_back(sample);
	    meanValid = false;
	    varianceValid = false;
	}
	void addSamples(SampleCollection newSamples);
	/**
	 * @brief Returns number of samples of the collection
	 * @return the number of samples
	 */
	unsigned long getNumOfSamples() const {
	    return samples.size();
	};
	/**
	 * @brief Gets the value of a sample in some position
	 * @param[in] position index of the sample in the vector of samples (beginning at 0)
	 * @return the value of the sample
	 */
	double getSample(unsigned long position) const {
	    return samples.at(position);
	}
	double getMean() ;
	double getVariance();
	std::unique_ptr<InfoItems> to_infoItems(unsigned int numDigitsPrec);
    private:
	void calcMean();
	void calcVariance();

	/// Flag that shows if stored mean value is valid (otherwise, it must be recalculated)
	bool meanValid = false;
	/// Flag that shows if stored variance value is valid (otherwise, it must be recalculated)
	bool varianceValid = false;
	/// Vector storing the samples
	std::vector<double> samples;
	/// Mean of stored samples
	double mean = 0.0;
	/// Variance of stored samples
	double variance = 0.0;
	/// Name of sample collection
	std::string sampleName = "";
	/// Smart pointer to object for configuring summary output of the sample collection
	std::shared_ptr<OutputConfigTraits> pOutputConfigTraits;
};

} /* namespace OpenCLIPER */

#endif /* INCLUDE_OPENCLIPER_SAMPLECOLLECTION_HPP_ */


