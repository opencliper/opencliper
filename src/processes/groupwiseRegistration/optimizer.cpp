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
 * optimizer.cpp
 *
 *  Created on: Apr 8, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/GroupwiseRegistration.hpp>

// Uncomment to show class-specific debug messages
#define OPTIMIZER_DEBUG

#if !defined NDEBUG && defined OPTIMIZER_DEBUG
    #define OPTIMIZER_CERR(x) CERR(x)
#else
    #define OPTIMIZER_CERR(x)
    #undef OPTIMIZER_DEBUG
#endif

namespace OpenCLIPER {

/**
 * @brief Performs optimization loop
 * @param[in] TData initial transformacion
 * @param[out] argsMC struct containing arguments needed in MotionCompensation
 */
void GroupwiseRegistration::optimizer(std::shared_ptr<Data> TData, ArgumentsMotionCompensation* argsMC) {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    OPTIMIZER_CERR("\n OPTIMIZER..." << std::endl);
    OPTIMIZER_CERR("===========================================\n" << std::endl);

    uint nFRAMES = getInput()->getTemporalDimSize(0);


    OPTIMIZER_CERR("\tCopying initial data to transformed data, just to initialize the object... ");
    transformedData = std::make_shared<XData>(getApp(), getInput(), false); // only copies dimensions and allocs memory
    OPTIMIZER_CERR("Done.");


    OPTIMIZER_CERR("\n\tAllocating memory... ");
    float* Hdif = new float[getParametersGD().nmax](); // Vector diferencia de costes
    float* Tdif = new float[getParametersGD().nmax](); // Vector diferencia transformacion
    OPTIMIZER_CERR("Done.");

    OPTIMIZER_CERR("\tCreating buffers... ");
    // XData containing the transformation matrix - T
    if(TData == nullptr) {
	std::vector<std::vector<dimIndexType>*>* pNDArrayDims = new std::vector<std::vector<dimIndexType>*>();
	pNDArrayDims->push_back(new std::vector<dimIndexType>({(uint)(ts->C[1][0] + ts->C[1][1] + 1), (uint)(ts->C[0][0] + ts->C[0][1] + 1), nFRAMES, ts->nt}));
	TData = std::make_shared<XData>(getApp(), pNDArrayDims, TYPEID_REAL); // only copies dimensions and allocs memory
        delete(pNDArrayDims);
        
        pInitZero->setOutput(TData);
        pInitZero->launch();
    }


    HData->device2Host();
    realType* Hbuffer = (realType*)(HData->getHostBuffer(0));

    DifData->device2Host();
    realType* Difbuffer = (realType*)(DifData->getHostBuffer(0));

    
    cl::Buffer Wnobj = cl::Buffer(getApp()->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float), &Wn, NULL);
    OPTIMIZER_CERR("Done." << std::endl);


    OPTIMIZER_CERR("\tSetting input in corresponding processes... ");
    pMetric->setInput(transformedData);
    pMetric->setOutput(VData);

    pCostReduction->setInput(VData);
    pCostReduction->setOutput(HData);

    pGradient->setInput(normInputData);
    pGradient->setOutput(dallData);

    pGradientMetric->setInput(transformedData);
    pGradientMetric->setOutput(dyData);

    pTransformationAux->setInput(TData);
    pTransformationAux->setOutput(TAuxData);

    pDeformation->setInput(TAuxData);
    pDeformation->setOutput(xnData);

    pDeformationAdjust->setOutput(xnData);

    pInitdx->setInput(dallData);
    pInitdx->setOutput(dxData);

    pGradientInterpolator->setInput(xnData);
    pGradientInterpolator->setOutput(dxData);

    pGradientRegularization->setInput(dtauallData);
    pGradientRegularization->setOutput(dthetaallData);

    pReduction7Dto6D->setInput(dthetaallData);
    pReduction7Dto6D->setOutput(auxData);

    pReduction6Dto5D->setInput(auxData);
    pReduction6Dto5D->setOutput(aux1Data);

    pReduction5Dto4D->setInput(aux1Data);
    pReduction5Dto4D->setOutput(aux2Data);

    pGradientJointAux->setInput(aux2Data);
    pGradientJointAux->setOutput(AuxData);

    pGradientWithSmoothTerms->setInput(AuxData);
    pGradientWithSmoothTerms->setOutput(dVData);

    pPermutedV->setInput(dVData);
    pPermutedV->setOutput(dVData);

    pAuxiliarMask->setInput(dVData);
    pAuxiliarMask->setOutput(XauxData);

    pAuxiliarCost->setInput(dVData);
    pAuxiliarCost->setOutput(cost1Data);

    pCost6DReduction->setInput(cost1Data);
    pCost6DReduction->setOutput(cost2Data);

    pCost6D->setInput(cost2Data);
    pCost6D->setOutput(dHData);

    pUpdateTransformation->setInput(dHData);
    pUpdateTransformation->setOutput(TData);

    pPermuteTAux->setInput(TAuxData);
    pPermuteTAux->setOutput(TAuxData);

    pShiftTAux->setInput(TAuxData);
    pShiftTAux->setOutput(TAuxData);

    pRegularization->setInput(TAuxData);
    pRegularization->setOutput(dtauallData);

    pInitIT->setInput(normInputData);
    pInitIT->setOutput(transformedData);

    pInterpolator->setInput(normInputData);
    pInterpolator->setOutput(transformedData);

    pSumLambdaMetric->setInput(dtauallData);
    pSumLambdaMetric->setOutput(VData);
    OPTIMIZER_CERR("Done." << std::endl);


    OPTIMIZER_CERR("\tSetting parameters... ");
    auto parTransformationAux = std::make_shared<TransformationAux::LaunchParameters>(ts->coef);
    pTransformationAux->setLaunchParameters(parTransformationAux);

    auto parDeformation = std::make_shared<Deformation::LaunchParameters>(ts->BBall);
    pDeformation->setLaunchParameters(parDeformation);

    auto parDeformationAdjust = std::make_shared<DeformationAdjust::LaunchParameters>(xData);
    pDeformationAdjust->setLaunchParameters(parDeformationAdjust);

    auto parGradientInterpolator = std::make_shared<GradientInterpolator::LaunchParameters>(dallData, xData);
    pGradientInterpolator->setLaunchParameters(parGradientInterpolator);

    auto parGradientRegularization = std::make_shared<GradientRegularization::LaunchParameters>(ts->coefg, ts->BBgall);
    pGradientRegularization->setLaunchParameters(parGradientRegularization);

    auto parReduction7Dto6D = std::make_shared<Reduction7Dto6D::LaunchParameters>(term.lambda[0], term.lambda[1], term.lambda[2], term.lambda[3]);
    pReduction7Dto6D->setLaunchParameters(parReduction7Dto6D);

    auto parReduction6Dto5D = std::make_shared<ReductionSum::LaunchParameters>(5);
    pReduction6Dto5D->setLaunchParameters(parReduction6Dto5D);

    auto parReduction5Dto4D = std::make_shared<ReductionSum::LaunchParameters>(2);
    pReduction5Dto4D->setLaunchParameters(parReduction5Dto4D);

    auto parGradientWithSmoothTerms = std::make_shared<GradientWithSmoothTerms::LaunchParameters>(dyData, dxData, ts->BBgall, ts->coefg);
    pGradientWithSmoothTerms->setLaunchParameters(parGradientWithSmoothTerms);

    auto parUpdateTransformation = std::make_shared<UpdateTransformation::LaunchParameters>(DifData, Wnobj);
    pUpdateTransformation->setLaunchParameters(parUpdateTransformation);

    auto parRegularization = std::make_shared<Regularization::LaunchParameters>(ts->BBall);
    pRegularization->setLaunchParameters(parRegularization);

    auto parSumLambdaMetric = std::make_shared<SumLambdaMetric::LaunchParameters>(term.lambda[0], term.lambda[1], term.lambda[2], term.lambda[3]);
    pSumLambdaMetric->setLaunchParameters(parSumLambdaMetric);

    auto parInterpolator = std::make_shared<Interpolator::LaunchParameters>(xnData, xData, bound_box);
    pInterpolator->setLaunchParameters(parInterpolator);
    OPTIMIZER_CERR("Done." << std::endl);


    OPTIMIZER_CERR("\tInitializing processes... ");
    pInterpolator->init();
    OPTIMIZER_CERR("Done." << std::endl);


    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Start the optimization  ---------------------------------------------------------------------
    ////////////////////////////////////////////////////////////////////////////////////////////////
    int iter = 0;
    
    pInitIT->launch();
    
    pInitZero->setOutput(dxData);
    pInitZero->launch();

    pInitZero->setOutput(dtauallData);
    pInitZero->launch();

    pMetric->launch();

    pCostReduction->setLaunchParameters(std::make_shared<CostReduction::LaunchParameters>(maskData, iter));
    pCostReduction->launch();
    HData->device2Host();

    OPTIMIZER_CERR("\n --> Initial cost = " << std::scientific << Hbuffer[0]);
    OPTIMIZER_CERR("\n --> Optimizing...\n" << std::endl);

    pGradient->launch();

    OPTIMIZER_CERR("  ================================================================================================== ");
    OPTIMIZER_CERR("\n |  n\t| Cost\t\tWn\t\tTdif\t\tHdif\t\t|" << std::endl);

    // Parameters needed for stop condition
    float p2 = getParametersGD().et * nFRAMES * ts->nt;
    float p4 = Hbuffer[0] * getParametersGD().eh;

    bool stop = false;
    iter = 1;

    std::shared_ptr<AuxiliarMask::LaunchParameters> parAuxiliarMask;
    std::shared_ptr<AuxiliarCost::LaunchParameters> parAuxiliarCost;
    std::shared_ptr<Cost6D::LaunchParameters> parCost6D;

    while(stop == false) {

	pInitZero->setOutput(TAuxData); // Initialize TAux
	pInitZero->launch();

	pTransformationAux->launch(); // Output TAux

	pInitZero->setOutput(xnData); // Initialize xn
	pInitZero->launch();

	pDeformation->launch();
	pDeformationAdjust->launch(); // Output xn

	pInitdx->launch();
	pGradientInterpolator->launch(); // Output dx
	pGradientMetric->launch(); // Output dy
	pGradientRegularization->launch(); // Output dtheta

	pReduction7Dto6D->launch(); // Output aux
	pReduction6Dto5D->launch(); // Output aux1
	pReduction5Dto4D->launch(); // Output aux2

	pGradientJointAux->launch(); // Output Aux
	pGradientWithSmoothTerms->launch(); // Output dV

	pPermutedV->launch();

	for(uint k = 0; k < dVData->getSpatialDimSize(3,0); k++) {
	    for(uint l = 0; l < dVData->getSpatialDimSize(4,0); l++) {

		parAuxiliarMask = std::make_shared<AuxiliarMask::LaunchParameters>(maskData, ts->coefg, l, k);
		pAuxiliarMask->setLaunchParameters(parAuxiliarMask);
		pAuxiliarMask->launch();    // Output XAux

		parAuxiliarCost = std::make_shared<AuxiliarCost::LaunchParameters>(XauxData, l, k);
		pAuxiliarCost->setLaunchParameters(parAuxiliarCost);
		pAuxiliarCost->launch();    // Output cost1

		pCost6DReduction->launch(); // Output cost2

		parCost6D = std::make_shared<Cost6D::LaunchParameters>(l, k);
		pCost6D->setLaunchParameters(parCost6D);
		pCost6D->launch();          // Output dH
	    }
	}

	// Projection of gradient
	pUpdateTransformation->launch();
	DifData->device2Host();
	pInitZero->setOutput(TAuxData); // Initialize TAux
	pInitZero->launch();

	pTransformationAux->launch(); // Output TAux

	pInitZero->setOutput(xnData); // Initialize xn
	pInitZero->launch();

	pDeformation->launch();
	pDeformationAdjust->launch(); // Output xn

	pPermuteTAux->launch();
	pShiftTAux->launch();

	pInitZero->setOutput(dtauallData); // Initialize every dtau
	pInitZero->launch();

	pRegularization->launch();

	pInitIT->launch();
	pInterpolator->launch();

	pMetric->launch();
	if(sumLambda() != 0) {
	    pSumLambdaMetric->launch();
	}

	pCostReduction->setLaunchParameters(std::make_shared<CostReduction::LaunchParameters>(maskData, iter));
	pCostReduction->launch();
	HData->device2Host();

	getApp()->getCommandQueue().enqueueReadBuffer(Wnobj, CL_TRUE, 0, sizeof(float), &Wn, NULL, NULL);
	evolution(Tdif, Hdif, &Wn, Difbuffer, Hbuffer, iter, flagW);
	getApp()->getCommandQueue().enqueueWriteBuffer(Wnobj, CL_TRUE, 0, sizeof(float), &Wn, NULL, NULL);

	OPTIMIZER_CERR("  -------------------------------------------------------------------------------------------------- " << std::endl);
	OPTIMIZER_CERR(" |  " << iter << "\t| " << Hbuffer[iter] << "\t" << Wn << "\t" << Tdif[iter - 1] << "\t" << Hdif[iter - 1] << "\t|\n");
	stop = stopCondition(Tdif[iter - 1], Hdif[iter - 1], p2, p4, getParametersGD().nmax, iter);
	iter = iter + 1;
    }
    OPTIMIZER_CERR("  ================================================================================================== " << std::endl);


    // Struct with necessary arguments to launch the motion compensation
    argsMC->xnData = xnData;
    argsMC->TData = TData;
    argsMC->xData = xData;
    argsMC->bound_box = bound_box;


    OPTIMIZER_CERR("\tReleasing memory... ");
    delete[] Hdif;
    delete[] Tdif;
    OPTIMIZER_CERR("Done.\n" << std::endl);

}
}

#undef OPTIMIZER_DEBUG
