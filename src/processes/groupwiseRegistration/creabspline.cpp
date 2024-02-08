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
 * creabspline.cpp
 *
 *  Created on: Apr 8, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/GroupwiseRegistration.hpp>

// Uncomment to show class-specific debug messages
#define CREABSPLINE_DEBUG

#if !defined NDEBUG && defined CREABSPLINE_DEBUG
    #define CREABSPLINE_CERR(x) CERR(x)
#else
    #define CREABSPLINE_CERR(x)
    #undef CREABSPLINE_DEBUG
#endif

namespace OpenCLIPER {

/**
 * @brief Stores the neccesary elements for the gradient and bspline transformation calculus,
 * B-spline product matrixes with the correspondent coefficients
 */
void GroupwiseRegistration::creabspline() {

    CREABSPLINE_CERR("\n\n CREATING B-SPLINES..." << std::endl);
    CREABSPLINE_CERR("===========================================\n" << std::endl);

    uint nCOLS = getInput()->getSpatialDimSize(COLUMNS, 0);
    
    xData->device2Host();
    dimIndexType* x = (dimIndexType*)(xData->getHostBuffer(0));
        
    // Buffer con contenido
    cl::Buffer cobj = cl::Buffer(getApp()->getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2 * sizeof(float), ts->c, NULL);
    cl::Buffer Dpobj = cl::Buffer(getApp()->getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2 * sizeof(int), ts->Dp, NULL);
 

    pControlPoints->setOutput(puData);

    pPermuteBB->setInput(auxBBallData);
    pPermuteBB->setOutput(ts->BBall);

    pRepmatBBg->setInput(auxBBgallData);
    pRepmatBBg->setOutput(aux1BBgallData);

    pPermuteBBg->setInput(aux1BBgallData);
    pPermuteBBg->setOutput(ts->BBgall);


    auto parControlPoints = std::make_shared<ControlPointsLocation::LaunchParameters>(ts->C[0][0], ts->C[1][0], cobj, Dpobj);
    pControlPoints->setLaunchParameters(parControlPoints);
    auto parPermuteBB = std::make_shared<PermuteBB::LaunchParameters>(Dpobj);
    pPermuteBB->setLaunchParameters(parPermuteBB);
    auto parCreateBBg = std::make_shared<CreateBBg::LaunchParameters>(ts->BBgall, Dpobj);
    pCreateBBg->setLaunchParameters(parCreateBBg);


    // ControlPointsLocation (pu) ---------------------------------------------------------
    // Localizaci�n de los puntos de control en p�xeles, a partir de los �ndices de
    // esos puntos de control
    pControlPoints->launch();

    // Create coef ------------------------------------------------------------------------
    // Coeficientes y matriz de productos para transformacion
    std::vector<realType>* coef = new std::vector<realType>(nCOLS*2*ts->nt);
    for(cl_uint b = 0; b < ts->coef->getSpatialDimSize(1,0); b++) { // min-max
	if(ts->E % 2 != 0) { // Impar - Orden 1 y 3
	    for(cl_uint i = 0; i < ts->coef->getSpatialDimSize(0,0); i++) {
		// Fila i, columna b, primera dimension - coef{1}
		coef->at(i + b * ts->coef->getSpatialDimSize(0,0) + 0 * ts->coef->getSpatialDimSize(0,0)*ts->coef->getSpatialDimSize(1,0)) = floor((x[0 + i * xData->getSpatialDimSize(0,0) + 0 * xData->getSpatialDimSize(0,0) * xData->getSpatialDimSize(1,0)] - ts->c[0]) / ts->Dp[0]) + ((ts->E + pow(-1, b + 1)) / 2) * pow(-1,
			b + 1);
		// Fila i, columna b, segunda dimension - coef{2}
		coef->at(i + b * ts->coef->getSpatialDimSize(0,0) + 1 * ts->coef->getSpatialDimSize(0,0)*ts->coef->getSpatialDimSize(1,0)) = floor((x[i + 0 * xData->getSpatialDimSize(0,0) + 1 * xData->getSpatialDimSize(0,0) * xData->getSpatialDimSize(1,0)] - ts->c[1]) / ts->Dp[1]) + ((ts->E + pow(-1, b + 1)) / 2) * pow(-1,
			b + 1);
	    }
	}
	else {   // Par - Orden 2
	    for(cl_uint i = 0; i < ts->coef->getSpatialDimSize(0,0); i++) {
		// Fila i, columna b, primera dimension - coef{1}
		coef->at(i + b * ts->coef->getSpatialDimSize(0,0) + 0 * ts->coef->getSpatialDimSize(0,0)*ts->coef->getSpatialDimSize(1,0)) = round((x[0 + i * xData->getSpatialDimSize(0,0) + 0 * xData->getSpatialDimSize(0,0) * xData->getSpatialDimSize(1,0)] - ts->c[0]) / ts->Dp[0]) + (ts->E / 2) * pow(-1, b + 1);
		// Fila i, columna b, segunda dimension - coef{2}
		coef->at(i + b * ts->coef->getSpatialDimSize(0,0) + 1 * ts->coef->getSpatialDimSize(0,0)*ts->coef->getSpatialDimSize(1,0)) = round((x[i + 0 * xData->getSpatialDimSize(0,0) + 1 * xData->getSpatialDimSize(0,0) * xData->getSpatialDimSize(1,0)] - ts->c[1]) / ts->Dp[1]) + (ts->E / 2) * pow(-1, b + 1);
	    }
	}
    }


    std::vector<realType>* BBAux = new std::vector<realType>(nCOLS*(ts->E + 1)*3*ts->nt);
    std::vector<dimIndexType> dataSize = {getInput()->getSpatialDimSize(COLUMNS, 0), getInput()->getSpatialDimSize(ROWS, 0)};
    // Matriz auxiliar para los coeficientes
    for(uint l = 0; l < ts->nt; l++) {
	int xbis[dataSize[l]];
	for(uint p = 0; p < dataSize[l]; p++) {
	    xbis[p] = p + 1;
	}
	// Margen para la imagen para la transformacion
	for(uint n = 0; n < dataSize[l]; n++) { // Filas
	    for(int k = coef->at(n + 0 * ts->coef->getSpatialDimSize(0,0) + l * ts->coef->getSpatialDimSize(0,0) * ts->coef->getSpatialDimSize(1,0)); k <= coef->at(n + 1 * ts->coef->getSpatialDimSize(0,0) + l * ts->coef->getSpatialDimSize(0,0)*ts->coef->getSpatialDimSize(1,0)); k++) { // Columnas
		BBAux->at(((k - 1) - coef->at(n + 0 * ts->coef->getSpatialDimSize(0,0) + l * ts->coef->getSpatialDimSize(0,0)*ts->coef->getSpatialDimSize(1,0)) + 1) + n * BBAuxData->getSpatialDimSize(1,0) + 0 * BBAuxData->getSpatialDimSize(1,0)*BBAuxData->getSpatialDimSize(0,0) + l * BBAuxData->getSpatialDimSize(1,0)*BBAuxData->getSpatialDimSize(0,0)*BBAuxData->getSpatialDimSize(2,0)) =
		    bsplineN(((xbis[n] - ts->c[l]) / ts->Dp[l]) - k, ts->E);
		BBAux->at(((k - 1) - coef->at(n + 0 * ts->coef->getSpatialDimSize(0,0) + l * ts->coef->getSpatialDimSize(0,0)*ts->coef->getSpatialDimSize(1,0)) + 1) + n * BBAuxData->getSpatialDimSize(1,0) + 1 * BBAuxData->getSpatialDimSize(1,0)*BBAuxData->getSpatialDimSize(0,0) + l * BBAuxData->getSpatialDimSize(1,0)*BBAuxData->getSpatialDimSize(0,0)*BBAuxData->getSpatialDimSize(2,0)) =
		    b1splineN(((xbis[n] - ts->c[l]) / ts->Dp[l]) - k, ts->E);
		BBAux->at(((k - 1) - coef->at(n + 0 * ts->coef->getSpatialDimSize(0,0) + l * ts->coef->getSpatialDimSize(0,0)*ts->coef->getSpatialDimSize(1,0)) + 1) + n * BBAuxData->getSpatialDimSize(1,0) + 2 * BBAuxData->getSpatialDimSize(1,0)*BBAuxData->getSpatialDimSize(0,0) + l * BBAuxData->getSpatialDimSize(1,0)*BBAuxData->getSpatialDimSize(0,0)*BBAuxData->getSpatialDimSize(2,0)) =
		    b2splineN(((xbis[n] - ts->c[l]) / ts->Dp[l]) - k, ts->E);

	    }
	}
    }
    std::vector<dimIndexType>* pDims = new std::vector<dimIndexType>({nCOLS, ts->E + 1, 3, ts->nt});
    NDArray* pBBAuxNDArray = new ConcreteNDArray<realType>(pDims, BBAux);
    BBAuxData = std::make_shared<XData>(getApp(), pBBAuxNDArray, TYPEID_REAL);

    
    pCreateBB->setInput(BBAuxData);
    pCreateBB->setOutput(auxBBallData);
    pCreateBB->launch();

    BBAuxData = nullptr;

    // Offset Coef ------------------------------------------------------------------------
     std::vector<realType>* coefbuffer = new std::vector<realType>(nCOLS*2*ts->nt);
    // Add offset to coeficients, used in transformation
    for(cl_uint i = 0; i < ts->coef->getSpatialDimSize(0,0); i++) {
	for(cl_uint b = 0; b < ts->coef->getSpatialDimSize(1,0); b++) { // min-max
	    for(cl_uint l = 0; l < ts->coef->getSpatialDimSize(1,0); l++) { // min-max
		coefbuffer->at(i + b * ts->coef->getSpatialDimSize(0,0) + l * ts->coef->getSpatialDimSize(0,0)*ts->coef->getSpatialDimSize(1,0)) = coef->at(i + b * ts->coef->getSpatialDimSize(0,0) + l * ts->coef->getSpatialDimSize(0,0) * ts->coef->getSpatialDimSize(1,0)) + 1 - coef->at((bound_box->at(l)) + 0 * ts->coef->getSpatialDimSize(0,0) + l *
			ts->coef->getSpatialDimSize(0,0) * ts->coef->getSpatialDimSize(1,0));
	    }
	}
    }
    pDims = new std::vector<dimIndexType>({nCOLS, 2, ts->nt});
    NDArray* pcoefNDArray = new ConcreteNDArray<realType>(pDims, coefbuffer);
    ts->coef = std::make_shared<XData>(getApp(), pcoefNDArray, TYPEID_REAL);
    

    // Permute BB ------------------------------------------------------------------------
    pPermuteBB->launch();
    auxBBallData = nullptr;

    // Coeficientes y productos matriciales para el c�lculo del gradiente
    // M�todo de coeficientes separados
    puData->device2Host();
    realType* pu = (realType*)(puData->getHostBuffer(0));
    std::vector<realType>* coefgbuffer = new std::vector<realType>((uint)(ts->C[1][0] + ts->C[1][1] + 1)*2*ts->nt);
    for(uint i = 0; i < ts->coefg->getSpatialDimSize(0,0); i++) {
	for(uint j = 0; j < ts->coefg->getSpatialDimSize(1,0); j++) { // min-max
	    coefgbuffer->at(i + j * ts->coefg->getSpatialDimSize(0,0) + 0 * ts->coefg->getSpatialDimSize(0,0)*ts->coefg->getSpatialDimSize(1,0)) = ceil(pu[i + 0 * puData->getSpatialDimSize(1,0) + 0 * puData->getSpatialDimSize(1,0) * puData->getSpatialDimSize(0,0)] - ceil(ts->rp[0])) + j * 2 * ceil(ts->rp[0]);
	    coefgbuffer->at(i + j * ts->coefg->getSpatialDimSize(0,0) + 1 * ts->coefg->getSpatialDimSize(0,0)*ts->coefg->getSpatialDimSize(1,0)) = ceil(pu[0 + i * puData->getSpatialDimSize(1,0) + 1 * puData->getSpatialDimSize(1,0) * puData->getSpatialDimSize(0,0)] - ceil(ts->rp[1])) + j * 2 * ceil(ts->rp[1]);
	}
    }

    int limxy[2] = {(ts->C[1][0] + ts->C[1][1] + 1), (ts->C[0][0] + ts->C[0][1] + 1)};
    float* xbis2 = new float[limxy[0] * 2];
    for(uint l = 0; l < 2; l++) {
	for(int i = 0; i < limxy[l]; i++) {
	    if(l == 0) {
		xbis2[i + 0 * limxy[l]] = pu[i + 0 * puData->getSpatialDimSize(1,0) + 0 * puData->getSpatialDimSize(1,0) * puData->getSpatialDimSize(0,0)];
	    }
	    else {
		xbis2[i + 1 * limxy[l]] = pu[0 + i * puData->getSpatialDimSize(1,0) + 1 * puData->getSpatialDimSize(1,0) * puData->getSpatialDimSize(0,0)];
	    }
	}
    }

    std::vector<realType>* BBAux1 = new std::vector<realType>((uint)(ts->C[1][0] + ts->C[1][1] + 1)* (uint)(2 * ceil(ts->rp[0]) + 1)*3*2);
    for(uint n = 0; n < ts->coefg->getSpatialDimSize(0,0); n++) {
	uint idxcoefg = n + 0 * ts->coefg->getSpatialDimSize(0,0) + 0 * ts->coefg->getSpatialDimSize(0,0) * ts->coefg->getSpatialDimSize(1,0);
	for(int k = coefgbuffer->at(n + 0 * ts->coefg->getSpatialDimSize(0,0) + 0 * ts->coefg->getSpatialDimSize(0,0) * ts->coefg->getSpatialDimSize(1,0)); k <= coefgbuffer->at(n + 1 * ts->coefg->getSpatialDimSize(0,0) + 0 * ts->coefg->getSpatialDimSize(0,0)*ts->coefg->getSpatialDimSize(1,0)); k++) {
	    BBAux1->at(((k - 1) - coefgbuffer->at(idxcoefg) + 1) + n * BBAux1Data->getSpatialDimSize(1,0) + 0 * BBAux1Data->getSpatialDimSize(1,0)*BBAux1Data->getSpatialDimSize(0,0) + 0 * BBAux1Data->getSpatialDimSize(1,0)*BBAux1Data->getSpatialDimSize(0,0)*BBAux1Data->getSpatialDimSize(2,0)) = bsplineN(((
			k - xbis2[n + 0 * limxy[0]]) / ts->Dp[0]), ts->E);
	    BBAux1->at(((k - 1) - coefgbuffer->at(idxcoefg) + 1) + n * BBAux1Data->getSpatialDimSize(1,0) + 0 * BBAux1Data->getSpatialDimSize(1,0)*BBAux1Data->getSpatialDimSize(0,0) + 1 * BBAux1Data->getSpatialDimSize(1,0)*BBAux1Data->getSpatialDimSize(0,0)*BBAux1Data->getSpatialDimSize(2,0)) = bsplineN(((
			k - xbis2[n + 1 * limxy[0]]) / ts->Dp[1]), ts->E);
	    BBAux1->at(((k - 1) - coefgbuffer->at(idxcoefg) + 1) + n * BBAux1Data->getSpatialDimSize(1,0) + 1 * BBAux1Data->getSpatialDimSize(1,0)*BBAux1Data->getSpatialDimSize(0,0) + 0 * BBAux1Data->getSpatialDimSize(1,0)*BBAux1Data->getSpatialDimSize(0,0)*BBAux1Data->getSpatialDimSize(2,0)) = b1splineN(((
			k - xbis2[n + 0 * limxy[0]]) / ts->Dp[0]), ts->E);
	    BBAux1->at(((k - 1) - coefgbuffer->at(idxcoefg) + 1) + n * BBAux1Data->getSpatialDimSize(1,0) + 1 * BBAux1Data->getSpatialDimSize(1,0)*BBAux1Data->getSpatialDimSize(0,0) + 1 * BBAux1Data->getSpatialDimSize(1,0)*BBAux1Data->getSpatialDimSize(0,0)*BBAux1Data->getSpatialDimSize(2,0)) = b1splineN(((
			k - xbis2[n + 1 * limxy[0]]) / ts->Dp[1]), ts->E);
	    BBAux1->at(((k - 1) - coefgbuffer->at(idxcoefg) + 1) + n * BBAux1Data->getSpatialDimSize(1,0) + 2 * BBAux1Data->getSpatialDimSize(1,0)*BBAux1Data->getSpatialDimSize(0,0) + 0 * BBAux1Data->getSpatialDimSize(1,0)*BBAux1Data->getSpatialDimSize(0,0)*BBAux1Data->getSpatialDimSize(2,0)) = b2splineN(((
			k - xbis2[n + 0 * limxy[0]]) / ts->Dp[0]), ts->E);
	    BBAux1->at(((k - 1) - coefgbuffer->at(idxcoefg) + 1) + n * BBAux1Data->getSpatialDimSize(1,0) + 2 * BBAux1Data->getSpatialDimSize(1,0)*BBAux1Data->getSpatialDimSize(0,0) + 1 * BBAux1Data->getSpatialDimSize(1,0)*BBAux1Data->getSpatialDimSize(0,0)*BBAux1Data->getSpatialDimSize(2,0)) = b2splineN(((
			k - xbis2[n + 1 * limxy[0]]) / ts->Dp[1]), ts->E);
	}
    }
    delete[] xbis2;

    pDims = new std::vector<dimIndexType>({(uint)(ts->C[1][0] + ts->C[1][1] + 1), (uint)(2 * ceil(ts->rp[0]) + 1), 3, 2});
    NDArray* pBBAux1NDArray = new ConcreteNDArray<realType>(pDims, BBAux1);
    std::shared_ptr<Data> BBAux1Data = std::make_shared<XData>(getApp(), pBBAux1NDArray, TYPEID_REAL);

    pCreateBBg->setInput(BBAux1Data);
    pCreateBBg->setOutput(auxBBgallData);
    pCreateBBg->launch();

    // Repmat BBg ------------------------------------------------------------------------
    pRepmatBBg->launch();

    // Permute BBg ------------------------------------------------------------------------
    pPermuteBBg->launch();
    aux1BBgallData = nullptr;

    // Modificacion de los coeficientes para evitar bordes
    for(uint i = 0; i < ts->coefg->getSpatialDimSize(0,0); i++) {
	for(uint j = 0; j < ts->coefg->getSpatialDimSize(1,0); j++) { // min-max
	    for(uint l = 0; l < ts->coefg->getSpatialDimSize(2,0); l++) {
		if(coefgbuffer->at(i + j * ts->coefg->getSpatialDimSize(0,0) + l * ts->coefg->getSpatialDimSize(0,0)*ts->coefg->getSpatialDimSize(1,0)) < 1)
		    coefgbuffer->at(i + j * ts->coefg->getSpatialDimSize(0,0) + l * ts->coefg->getSpatialDimSize(0,0)*ts->coefg->getSpatialDimSize(1,0)) = 1;
		if(coefgbuffer->at(i + j * ts->coefg->getSpatialDimSize(0,0) + l * ts->coefg->getSpatialDimSize(0,0)*ts->coefg->getSpatialDimSize(1,0)) > (int)dataSize[l])
		    coefgbuffer->at(i + j * ts->coefg->getSpatialDimSize(0,0) + l * ts->coefg->getSpatialDimSize(0,0)*ts->coefg->getSpatialDimSize(1,0)) = dataSize[l];
	    }
	}
    }
    pDims = new std::vector<dimIndexType>({(uint)(ts->C[1][0] + ts->C[1][1] + 1),2,ts->nt});
    NDArray* pcoefgNDArray = new ConcreteNDArray<realType>(pDims, coefgbuffer);
    ts->coefg = std::make_shared<XData>(getApp(), pcoefgNDArray, TYPEID_REAL);
    
    
    delete(coef);
    delete(BBAux);
    delete(BBAux1);

    CREABSPLINE_CERR(" Done.\n" << std::endl);
}

}

#undef CREABSPLINE_DEBUG
