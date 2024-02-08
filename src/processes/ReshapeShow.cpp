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
 * ReshapeShow.cpp
 *
 *  Created on: 18 de mar. de 2019
 *      Author: fedsim
 */

#include <OpenCLIPER/processes/ReshapeShow.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <OpenCLIPER/XData.hpp>
#include <OpenCLIPER/hostKernelFunctions.hpp>

namespace OpenCLIPER {

void ReshapeShow::init() {
    auto pIP = std::dynamic_pointer_cast<InitParameters>(pInitParameters);
    if(!pIP) pIP = std::unique_ptr<InitParameters>(new InitParameters());
    timeDimension = pIP->timeDimension;

    if(!getInput())
	throw(CLError(CL_INVALID_MEM_OBJECT, "OpenCLIPER::ReshapeShow::init(): init() called before setInputData()"));

    if(!getInput()->getAllSizesEqual())
	throw(std::invalid_argument("OpenCLIPER::ReshapeShow::init(): ReshapeShow for variable-size data objects is not implemented at this time"));

    // Get number of tiles to paint: the size of the 3rd generic dimension (might be the 3rd spatial dimension or the coil dimension).
    dimIndexType nTiles = getInput()->getDimSize(2, 0);

    // Guess best window distribution for given number of tiles (best fit to 16/9)
    unsigned bestTilesH = 0, bestTilesV = 0; // Initialized to shut up a warning about uninitialized use
    float bestFit = MAXFLOAT;
    unsigned nTilesH = nTiles;
    unsigned nTilesV = 1;
    do {
	nTilesH = ceil((float)nTiles / (float)nTilesV);
	float ratio = (float)nTilesH / (float)nTilesV;
	float fit = abs(16.0/9.0 - ratio);

	if(fit < bestFit) {
	    bestTilesH = nTilesH;
	    bestTilesV = nTilesV;
	    bestFit = fit;
	}

	++nTilesV;
    }
    while(nTilesV <= nTilesH);

    // Get slice and mosaic window dimensions
    uint sliceWidth = getInput()->getSpatialDimSize(0, 0);
    uint sliceHeight = getInput()->getSpatialDimSize(1, 0);
    uint nFrames = getInput()->getTemporalDimSize(timeDimension);
    winWidth = bestTilesH * sliceWidth;
    winHeight = bestTilesV * sliceHeight;

    auto pSpatialDims = new std::vector<dimIndexType> ({winWidth, winHeight});
    auto pTemporalDims = new std::vector<dimIndexType> (1, nFrames);

    // Calculate coordinates for the upper-left corner of each tile (to show text, etc).
    uint tileX=0, tileY=0;
    tileCoords.resize(nTiles);
    for(uint i=0; i<nTiles; i++) {
	tileCoords[i]={tileX,tileY};

	tileX+=sliceWidth;
	if(tileX>=winWidth) {
	    tileY+=sliceHeight;
	    tileX=0;
	}
    }

    canvas = std::make_shared<XData>(getApp(), nFrames, pSpatialDims, pTemporalDims, TYPEID_REAL);

    kernel = getApp()->getKernel("reshape_show");
}

void ReshapeShow::launch() {
    cl::NDRange globalSizes = cl::NDRange(winWidth, winHeight);
    cl::NDRange localSizes = cl::NDRange();

    kernel.setArg(0, *canvas->getDeviceBuffer());
    kernel.setArg(1, *getInput()->getDeviceBuffer());
    kernel.setArg(2, winWidth);
    kernel.setArg(3, timeDimension);

    cl_int err;
    if((err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSizes, localSizes, NULL, NULL)) != CL_SUCCESS)
	throw(CLError(err, getApp()->getOpenCLErrorCodeStr(err)));
}

} // namespace OpenCLIPER
