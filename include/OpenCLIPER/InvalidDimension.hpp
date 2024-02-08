/*
 * Invaliddimension.h
 *
 *  Created on: Apr 28, 2020
 *      Author: manrod
 */

#ifndef SRC_INVALIDDIMENSION_H_
#define SRC_INVALIDDIMENSION_H_

#include <stdexcept>
#include<OpenCLIPER/defs.hpp>

namespace OpenCLIPER {

class InvalidDimension: public std::out_of_range {
public:
	explicit InvalidDimension (const std::string& what_arg, dimIndexType invalidDimId, dimIndexType wrongValue,
			dimIndexType rightValue);
	virtual ~InvalidDimension();
	dimIndexType getInvalidDimId() {return invalidDimId;};
	dimIndexType getWrongValue() {return wrongValue;};
	dimIndexType getRightValue() {return rightValue;};

private:
	dimIndexType invalidDimId, wrongValue, rightValue;
};

} /* namespace OpenCLIPER */

#endif /* SRC_INVALIDDIMENSION_H_ */
