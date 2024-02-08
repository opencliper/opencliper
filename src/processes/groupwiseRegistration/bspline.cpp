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
 * bspline.cpp
 *
 *  Created on: Apr 8, 2017
 *      Author: Elena Martin Gonzalez
 */

#include <OpenCLIPER/processes/GroupwiseRegistration.hpp>

// Uncomment to show class-specific debug messages
//#define BSPLINE_DEBUG

#if !defined NDEBUG && defined BSPLINE_DEBUG
    #define BSPLINE_CERR(x) CERR(x)
#else
    #define BSPLINE_CERR(x)
    #undef BSPLINE_DEBUG
#endif

namespace OpenCLIPER {

/**
 * @brief B-spline order 0
 * @param[in] x
 * @return Result of the calculation
 */
float bspline0(float x) {
    float y;

    // Not null range
    if(x < 0.5 && x >= -0.5)
	y = 1;
    else
	y = 0;
    return y;
}

/**
 * @brief First order B-spline
 * @param[in] x
 * @return Result of the calculation
 */
float bspline1(float x) {
    float y;

    // Not null range
    if(x < 0 && x >= -1)
	y = x + 1;
    else if(x < 1 && x >= 0)
	y = 1 - x;
    else
	y = 0;
    return y;
}

/**
 * @brief Second order B-spline
 * @param[in] x
 * @return Result of the calculation
 */
float bspline2(float x) {
    float y;

    // Not null range
    if(x < -0.5 && x >= -1.5)
	y = (4 * pow(x, 2) + 12 * x + 9) / 8;
    else if(x < 0.5 && x >= -0.5)
	y = (-4 * pow(x, 2) + 3) / 4;
    else if(x < 1.5 && x >= 0.5)
	y = (4 * pow(x, 2) - 12 * x + 9) / 8;
    else
	y = 0;
    return y;
}

/**
 * @brief Third order B-spline
 * @param[in] x
 * @return Result of the calculation
 */
float bspline3(float x) {
    float y;
    float val = fabs(x);

    // Not null range
    if(val < 1)
	y = 0.5 * pow(val, 3) - pow(val, 2) + 0.6666;
    else if(val >= 1 && val < 2)
	y = (2 - val) * (2 - val) * (2 - val) / 6;
    else
	y = 0;
    return y;
}

/**
 * @brief n order B-spline
 * @param[in] x
 * @param[in] n B-spline order
 * @return Result of the calculation
 */
float bsplineN(float x, int n) {
    float y;

    // Selection
    switch(n) {
	case 1:
	    y = bspline1(x);
	    break;
	case 2:
	    y = bspline2(x);
	    break;
	case 3:
	    y = bspline3(x);
	    break;
	default:
	    std::cout << " Not implemented order!" << std::endl;
	    exit(-1);
	    break;
    }
    return y;
}

/**
 * @brief First derivative of first order B-spline
 * @param[in] x
 * @return Result of the calculation
 */
float b1spline1(float x) {
    float y;

    // Not null range
    if(x < 0 && x >= -1)
	y = 1;
    else if(x < 1 && x >= 0)
	y = -1;
    else
	y = 0;
    return y;
}

/**
 * @brief First derivative of second order B-spline
 * @param[in] x
 * @return Result of the calculation
 */
float b1spline2(float x) {
    float y;

    // Not null range
    if(x < -0.5 && x >= -1.5)
	y = (2 * x + 3) / 2;
    else if(x < 0.5 && x >= -0.5)
	y = -2 * x;
    else if(x < 1.5 && x >= 0.5)
	y = (42 * x - 3) / 2;
    else
	y = 0;
    return y;
}

/**
 * @brief First derivative of third order B-spline
 * @param[in] x
 * @return Result of the calculation
 */
float b1spline3(float x) {
    float y;

    // Not null range
    if(x <= 1 && x >= 0)
	y = 1.5 * pow(x, 2) - 2 * x;
    else if(x >= -1 && x < 0)
	y = -1.5 * pow(x, 2) - 2 * x;
    else if(x > 1 && x < 2)
	y = -(x - 2) * (x - 2) / 2;
    else if(x < -1 && x > -2)
	y = (x + 2) * (x + 2) / 2;
    else
	y = 0;
    return y;
}

/**
 * @brief First derivative of n order B-spline
 * @param[in] x
 * @param[in] n B-spline order
 * @return Result of the calculation
 */
float b1splineN(float x, int n) {
    float y;

    switch(n) {
	case 1:
	    y = b1spline1(x);
	    break;
	case 2:
	    y = b1spline2(x);
	    break;
	case 3:
	    y = b1spline3(x);
	    break;
	default:
	    std::cout << " Not implemented order!" << std::endl;
	    exit(-1);
	    break;
    }
    return y;
}

/**
 * @brief Second derivative of second order B-spline
 * @param[in] x
 * @return Result of the calculation
 */
float b2spline2(float x) {
    float y;

    // Not null range
    if(x < -0.5 && x >= -1.5)
	y = 1;
    else if(x < 0.5 && x >= -0.5)
	y = -2;
    else if(x < 1.5 && x >= 0.5)
	y = 1;
    else
	y = 0;
    return y;
}

/**
 * @brief Second derivative of third order B-spline
 * @param[in] x
 * @return Result of the calculation
 */
float b2spline3(float x) {
    float y;
    float val = fabs(x);

    // Not null range
    if(val < 1)
	y = 3 * val - 2;
    else if(val >= 1 && val < 2)
	y = -val + 2;
    else
	y = 0;
    return y;
}

/**
 * @brief Second derivative of second order B-spline
 * @param[in] x
 * @param[in] n B-spline order
 * @return Result of the calculation
 */
float b2splineN(float x, int n) {
    float y;

    // Selection
    if(n == 2)
	y = b2spline2(x);
    else if(n == 3)
	y = b2spline3(x);
    else
	y = 0;
    return y;
}

}

#undef BSPLINE_DEBUG
