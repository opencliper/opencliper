/*
 * Invaliddimension.cpp
 *
 *  Created on: Apr 28, 2020
 *      Author: manrod
 */

#include <OpenCLIPER/InvalidDimension.hpp>

namespace OpenCLIPER {

InvalidDimension::InvalidDimension (const std::string& what_arg, dimIndexType invalidDimId, dimIndexType wrongValue,
		dimIndexType rightValue) : std::out_of_range(what_arg) {
	this->invalidDimId = invalidDimId;
	this->wrongValue = wrongValue;
	this->rightValue = rightValue;
}
InvalidDimension::~InvalidDimension() {
	// TODO Auto-generated destructor stub
}

} /* namespace OpenCLIPER */
