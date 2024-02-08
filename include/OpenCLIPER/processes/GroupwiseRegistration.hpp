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

#ifndef GROUPWISEREGISTRATION_HPP
#define GROUPWISEREGISTRATION_HPP
#include <OpenCLIPER/CLapp.hpp>
#include <OpenCLIPER/Process.hpp>
#include <OpenCLIPER/ConcreteNDArray.hpp>
#include <OpenCLIPER/XData.hpp>
#include <OpenCLIPER/Data.hpp>

#include <OpenCLIPER/processes/groupwiseRegistration/reductions/Reduction7Dto6D.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/reductions/Cost6DReduction.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/reductions/CostReduction.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/reductions/ReductionSum.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/creabspline/ControlPointsLocation.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/creabspline/CreateBB.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/creabspline/PermuteBB.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/creabspline/CreateBBg.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/creabspline/RepmatBBg.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/creabspline/PermuteBBg.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/initialize/InitZero.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/initialize/InitZeroRect.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/initialize/CopyDataGPU.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/initialize/Initdx.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/initialize/DataNormalization.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/AuxiliarMask.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/AuxiliarCost.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/Metric.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/SumLambdaMetric.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/Gradient.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/TransformationAux.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/Deformation.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/DeformationAdjust.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/GradientInterpolator.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/GradientMetric.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/GradientRegularization.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/GradientJointAux.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/PermutedV.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/UpdateTransformation.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/PermuteTAux.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/ShiftTAux.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/Regularization.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/Interpolator.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/Cost6D.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/Regularization.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/GradientRegularization.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/GradientWithSmoothTerms.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/AdjointInterpolator.hpp>

#include <algorithm>
#include <numeric> // Needed for gcc 7 (default compiler in Ubuntu 18.04)
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>


namespace OpenCLIPER {

/**
 * @brief Structure with arguments obtained in Groupwise Registration, needed in Motion Compensation
 *
 */
struct ArgumentsMotionCompensation {
    std::shared_ptr<Data> xnData;
    std::shared_ptr<Data> TData;
    std::shared_ptr<Data> xData;
    std::vector<dimIndexType>* bound_box;
};

/**
 * @brief Struct with host memory objects used in creabspline()
 *
 */
struct Bspline {
    uint nt;
    int Dp[2];
    uint E;
    float c[2];
    float rp[2];
    int C[2][2];
    std::shared_ptr<Data> coef;
    std::shared_ptr<Data> coefg;
    std::shared_ptr<Data> BBgall;
    std::shared_ptr<Data> BBall;
};

/**
 * @brief Struct with the corresponding weights applied to the smoothness/regularization terms.
 *
 */
struct Terms {
    std::vector<float> lambda; //!< Smoothness terms (1st spatial, 2nd spatial, 1st temporal, 2nd temporal)
};

/**
 * @brief Struct with the parameters used in gradient descent optimizer.
 *
 */
struct ParametersGD {
    uint nmax = 0; //!< Maximum number of optimization loop iterations
    float et = 0.0; //!< Transformation norm threshold (in pixels)
    float eh = 0.0; //!< Metric variation threshold (e.g. 0.005 => 0.5% of initial metric)
};


/**
 * @brief Process class to perform the groupwise registration, with metric VI (variance of intensity).
 *
 */
class GroupwiseRegistration: public Process {
    public:
        
        struct InitParameters: ProcessCore::InitParameters {
	    float W;
            bool flagW;
            uint radius;
	    uint E;
	    int* Dp;
            uint nmax; //!< Maximum number of optimization loop iterations
	    float et; //!< Transformation norm threshold (in pixels)
	    float eh; //!< Metric variation threshold (e.g. 0.005 => 0.5% of initial metric)
	    std::vector<realType> lambda; //!< Smoothness terms (1st spatial, 2nd spatial, 1st temporal, 2nd temporal)

	    /// constructor
	    explicit InitParameters(float W, bool flagW, uint radius, uint E, int* Dp, uint nmax, float et, float eh, const std::vector<realType>& lambda): W(W), flagW(flagW), radius(radius), E(E), Dp(Dp), nmax(nmax), et(et), eh(eh), lambda(lambda) {}
	};
        
        
        struct LaunchParameters: Process::LaunchParameters {
	    ArgumentsMotionCompensation* argsMC;
	    std::shared_ptr<Data> TData;
	    int currentMotionIter;

	    LaunchParameters(ArgumentsMotionCompensation* args, const std::shared_ptr<Data>& TData, int iter):
		argsMC(args), TData(TData), currentMotionIter(iter) {}
	};

	void init();
	void launch();

	// Getters
	/** Returns struct containig weights for every regularization term
	 * @returns weights for every regularization term
	 */
	Terms getTerms() {
	    return term;
	};
	/** Returns struct containing parameters needed in Gradient Descent Algorithm
	 * @returns parameters needed in Gradient Descent Algorithm
	 */
	ParametersGD getParametersGD() {
	    return param;
	};

	// Setters
	void setTerms(const std::vector<float> &lambda);
	void setParametersGD(const unsigned int paramNmax, const float paramEt, const float paramEh);

    private:
	// We need to create subprocesses, so can't just inherit out parent class' constructors
	GroupwiseRegistration(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP = nullptr);
	GroupwiseRegistration(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data> pIn, const std::shared_ptr<Data> pOut, const std::shared_ptr<ProfileParameters>& pPP = nullptr);

	// We must allow our constructors to be called from Process::create()
	friend std::shared_ptr<GroupwiseRegistration> Process::create<GroupwiseRegistration>(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP);
	friend std::shared_ptr<GroupwiseRegistration> Process::create<GroupwiseRegistration>(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data> pIn, const std::shared_ptr<Data> pOut, const std::shared_ptr<ProfileParameters>& pPP);

	// Internal methods
	float sumLambda();
	int setMask(std::vector<dimIndexType>* X, std::vector<dimIndexType>* bound_box, const int radius);
	void creabspline();
	void optimizer(std::shared_ptr<Data> TData, ArgumentsMotionCompensation* argsMC);
	void evolution(float* Tdif, float* Hdif, float* Wn, realType* Dif, realType* H, int iter, bool flagW);
	bool stopCondition(const float p1, const float p3, const float p2, const float p4, const int p5, const int p6);

	// Attributes
	Terms term; // Lambda terms
	ParametersGD param; // Parameters for Gradient Descent Algorithm
	std::shared_ptr<Bspline> ts;

        
	std::shared_ptr<Process> pControlPoints;
	std::shared_ptr<Process> pCreateBB;
	std::shared_ptr<Process> pPermuteBB;
	std::shared_ptr<Process> pCreateBBg;
	std::shared_ptr<Process> pRepmatBBg;
	std::shared_ptr<Process> pPermuteBBg;

        std::shared_ptr<Process> pInitZero;
	std::shared_ptr<Process> pInitIT;
	std::shared_ptr<Process> pMetric;
	std::shared_ptr<Process> pSumLambdaMetric;
	std::shared_ptr<Process> pGradient;
	std::shared_ptr<Process> pPermutedV;
	std::shared_ptr<Process> pUpdateTransformation;
	std::shared_ptr<Process> pCost6D;
	std::shared_ptr<Process> pGradientMetric;
	std::shared_ptr<Process> pGradientJointAux;
	std::shared_ptr<Process> pGradientWithSmoothTerms;
	std::shared_ptr<Process> pReduction7Dto6D;
	std::shared_ptr<Process> pReduction6Dto5D;
	std::shared_ptr<Process> pReduction5Dto4D;
	std::shared_ptr<Process> pAuxiliarMask;
	std::shared_ptr<Process> pAuxiliarCost;
	std::shared_ptr<Process> pCost6DReduction;
	std::shared_ptr<Process> pCostReduction;
	std::shared_ptr<Process> pTransformationAux;
	std::shared_ptr<Process> pDeformation;
	std::shared_ptr<Process> pDeformationAdjust;
	std::shared_ptr<Process> pInitdx;
	std::shared_ptr<Process> pGradientInterpolator;
	std::shared_ptr<Process> pPermuteTAux;
	std::shared_ptr<Process> pShiftTAux;
	std::shared_ptr<Process> pRegularization;
	std::shared_ptr<Process> pInterpolator;
	std::shared_ptr<Process> pGradientRegularization;
        std::shared_ptr<Process> pDataNormalization;
        
        std::shared_ptr<Data> normInputData;
        std::shared_ptr<Data> transformedData;
	std::shared_ptr<Data> maskData;
        std::shared_ptr<Data> xData;
        std::shared_ptr<Data> puData;
        std::shared_ptr<Data> BBAuxData;
        std::shared_ptr<Data> BBAux1Data;
        std::shared_ptr<Data> auxBBallData;
        std::shared_ptr<Data> auxBBgallData;
        std::shared_ptr<Data> aux1BBgallData;
        std::shared_ptr<Data> TAuxData;
        std::shared_ptr<Data> VData;
        std::shared_ptr<Data> dtauallData;
        std::shared_ptr<Data> dthetaallData;
        std::shared_ptr<Data> HData;
        std::shared_ptr<Data> dallData;
        std::shared_ptr<Data> dxData;
        std::shared_ptr<Data> dyData;
        std::shared_ptr<Data> xnData;
        std::shared_ptr<Data> auxData;
        std::shared_ptr<Data> aux1Data;
        std::shared_ptr<Data> aux2Data;
        std::shared_ptr<Data> AuxData;
        std::shared_ptr<Data> dVData;
        std::shared_ptr<Data> XauxData;
        std::shared_ptr<Data> cost1Data;
        std::shared_ptr<Data> cost2Data;
        std::shared_ptr<Data> dHData;
        std::shared_ptr<Data> DifData;
        
        
        std::vector<dimIndexType>* bound_box;
        float Wn;
        bool flagW;
        dimIndexType longr1;
        dimIndexType longr2;
        dimIndexType longr1margin;
        dimIndexType longr2margin;

};



/**
 * @brief B-spline order 0
 * @param[in] x
 * @return Result of the calculation
 */
float bspline0(float x);

/**
 * @brief First order B-spline
 * @param[in] x
 * @return Result of the calculation
 */
float bspline1(float x);

/**
 * @brief Second order B-spline
 * @param[in] x
 * @return Result of the calculation
 */
float bspline2(float x);

/**
 * @brief Third order B-spline
 * @param[in] x
 * @return Result of the calculation
 */
float bspline3(float x);

/**
 * @brief n order B-spline
 * @param[in] x
 * @param[in] n B-spline order
 * @return Result of the calculation
 */
float bsplineN(float x, int n);

/**
 * @brief First derivative of first order B-spline
 * @param[in] x
 * @return Result of the calculation
 */
float b1spline1(float x);

/**
 * @brief First derivative of second order B-spline
 * @param[in] x
 * @return Result of the calculation
 */
float b1spline2(float x);

/**
 * @brief First derivative of third order B-spline
 * @param[in] x
 * @return Result of the calculation
 */
float b1spline3(float x);

/**
 * @brief First derivative of n order B-spline
 * @param[in] x
 * @param[in] n B-spline order
 * @return Result of the calculation
 */
float b1splineN(float x, int n);

/**
 * @brief Second derivative of second order B-spline
 * @param[in] x
 * @return Result of the calculation
 */
float b2spline2(float x);

/**
 * @brief Second derivative of third order B-spline
 * @param[in] x
 * @return Result of the calculation
 */
float b2spline3(float x);

/**
 * @brief Second derivative of second order B-spline
 * @param[in] x
 * @param[in] n B-spline order
 * @return Result of the calculation
 */
float b2splineN(float x, int n);

}

#endif // GROUPWISEREGISTRATION_HPP
