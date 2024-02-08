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

#include <OpenCLIPER/processes/GroupwiseRegistration.hpp>
#define GROUPWISEREGISTRATION_DEBUG

#if !defined NDEBUG && defined GROUPWISEREGISTRATION_DEBUG
    #define GROUPWISEREGISTRATION_CERR(x) CERR(x)
#else
    #define GROUPWISEREGISTRATION_CERR(x)
    #undef GROUPWISEREGISTRATION_DEBUG
#endif

namespace OpenCLIPER {

GroupwiseRegistration::GroupwiseRegistration(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP): Process(pCLapp, pPP) {
    // Create subprocess objects
    pDataNormalization = Process::create<DataNormalization>(getApp());
    
    pControlPoints = Process::create<ControlPointsLocation>(pCLapp);
    pCreateBB = Process::create<CreateBB>(pCLapp);
    pPermuteBB = Process::create<PermuteBB>(pCLapp);
    pCreateBBg = Process::create<CreateBBg>(pCLapp);
    pRepmatBBg = Process::create<RepmatBBg>(pCLapp);
    pPermuteBBg = Process::create<PermuteBBg>(pCLapp);

    pInitZero = Process::create<InitZero>(pCLapp);
    pInitIT = Process::create<CopyDataGPU>(pCLapp);
    pMetric = Process::create<Metric>(pCLapp);
    pSumLambdaMetric = Process::create<SumLambdaMetric>(pCLapp);
    pGradient = Process::create<Gradient>(pCLapp);
    pPermutedV = Process::create<PermutedV>(pCLapp);
    pUpdateTransformation = Process::create<UpdateTransformation>(pCLapp);
    pCost6D = Process::create<Cost6D>(pCLapp);
    pGradientMetric = Process::create<GradientMetric>(pCLapp);
    pGradientJointAux = Process::create<GradientJointAux>(pCLapp);
    pGradientWithSmoothTerms = Process::create<GradientWithSmoothTerms>(pCLapp);
    pReduction7Dto6D = Process::create<Reduction7Dto6D>(pCLapp);
    pReduction6Dto5D = Process::create<ReductionSum>(pCLapp);
    pReduction5Dto4D = Process::create<ReductionSum>(pCLapp);
    pAuxiliarMask = Process::create<AuxiliarMask>(pCLapp);
    pAuxiliarCost = Process::create<AuxiliarCost>(pCLapp);
    pCost6DReduction = Process::create<Cost6DReduction>(pCLapp);
    pCostReduction = Process::create<CostReduction>(pCLapp);
    pTransformationAux = Process::create<TransformationAux>(pCLapp);
    pDeformation = Process::create<Deformation>(pCLapp);
    pDeformationAdjust = Process::create<DeformationAdjust>(pCLapp);
    pInitdx = Process::create<Initdx>(pCLapp);
    pGradientInterpolator = Process::create<GradientInterpolator>(pCLapp);
    pPermuteTAux = Process::create<PermuteTAux>(pCLapp);
    pShiftTAux = Process::create<ShiftTAux>(pCLapp);
    pRegularization = Process::create<Regularization>(pCLapp);
    pInterpolator = Process::create<Interpolator>(pCLapp);
    pGradientRegularization = Process::create<GradientRegularization>(pCLapp);
}

GroupwiseRegistration::GroupwiseRegistration(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data> pIn, const std::shared_ptr<Data> pOut, const std::shared_ptr<ProfileParameters>& pPP): GroupwiseRegistration(pCLapp, pPP) {
    // Set input/output as given
    setInput(pIn);
    setOutput(pOut);
}

void GroupwiseRegistration::init() {
    auto pIP = std::dynamic_pointer_cast<InitParameters>(pInitParameters);
    if(!getInput())
	BTTHROW(CLError(CL_INVALID_MEM_OBJECT, "init() called before setInputData()"), "GroupwiseRegistration::init");

    if(!getInput()->getAllSizesEqual())
	BTTHROW(std::invalid_argument("Something went wrong with the input data"), "GroupwiseRegistration::init");
    

    uint nROWS = getInput()->getSpatialDimSize(ROWS, 0);
    uint nCOLS = getInput()->getSpatialDimSize(COLUMNS, 0);
    uint nFRAMES = getInput()->getTemporalDimSize(0);
    

    normInputData = std::make_shared<XData>(getApp(), getInput(), false); // Normalized data (only real part)
    transformedData = std::make_shared<XData>(getApp(), getInput(), false); // only copies dimensions and allocs memory
    
    bound_box = new std::vector<dimIndexType>(4);
    std::vector<dimIndexType>* mask = new std::vector<dimIndexType>(nROWS*nCOLS); // Region of interest - ROI 
    int areaX = 0;
    if(pIP->radius != 0) { // ROI circular
	areaX = setMask(mask, bound_box, pIP->radius);

    } else { // ROI wholebox
	int Nh = pIP->Dp[0] * (pIP->E + 1);
	int Nv = pIP->Dp[1] * (pIP->E + 1);
	for(dimIndexType i = Nv; i < nROWS - Nv; i++) { // Rows
	    for(dimIndexType j = Nh; j < nCOLS - Nh; j++) { // Columns
		mask->at(j + i * nCOLS) = 1;
		areaX++;
	    }
	}
	bound_box->at(0) = Nv + 1;
	bound_box->at(1) = nROWS - Nv;
	bound_box->at(2) = Nh + 1;
	bound_box->at(3) = nCOLS - Nh;
    }
    std::vector<dimIndexType>* pDims = new std::vector<dimIndexType>({nROWS, nROWS});
    NDArray* pmaskNDArray = new ConcreteNDArray<dimIndexType>(pDims, mask);
    maskData = std::make_shared<XData>(getApp(), pmaskNDArray, TYPEID_INDEX);
    delete(mask);
    
    longr1 = bound_box->at(2) - bound_box->at(0) + 1;
    longr2 = bound_box->at(3) - bound_box->at(1) + 1;

    // Hay que meter una comprobaci�n de que no se salga de los l�mites de la imagen cuando se a�ade el margen.
    std::vector<dimIndexType>* margin = new std::vector<dimIndexType>({(uint)ceil(nCOLS / 40), (uint)ceil(nROWS / 40), (uint)ceil(nFRAMES / 40)});
    longr1margin = (bound_box->at(3) + margin->at(0) + 1) - (bound_box->at(1) - margin->at(0) - 1) + 1;
    longr2margin = (bound_box->at(2) + margin->at(1) + 1) - (bound_box->at(0) - margin->at(1) - 1) + 1;

    
    Wn = pIP->W / areaX; // Weight normalization
    flagW = pIP->flagW; // Adaptive weight
    
     // Parameters for gradient descent optimization (Max number of iterations for the optimization loop, 0.01 pixels, 0.5% of initial)
    setParametersGD(pIP->nmax, pIP->et, pIP->eh);
    // Set spatio-temporal weigths (1st spatial, 2nd spatial, 1st temp, 2nd temp)
    setTerms(pIP->lambda);

    
    
    // Compute x, the original mesh -----------------------------------------------------------------------------
    std::vector<dimIndexType>* x = new std::vector<dimIndexType>(nROWS*nCOLS*2); 
    for(dimIndexType i = 0; i < nCOLS; i++) {
	for(dimIndexType j = 0; j < nROWS; j++) {
	    x->at(i + j * nCOLS + 0 * nROWS*nCOLS) = j + 1;
	    x->at(i + j * nCOLS + 1 * nROWS*nCOLS) = i + 1;
	}
    }
    pDims = new std::vector<dimIndexType>({nROWS, nROWS, 2});
    NDArray* pxNDArray = new ConcreteNDArray<dimIndexType>(pDims, x);
    xData = std::make_shared<XData>(getApp(), pxNDArray, TYPEID_INDEX);
    delete(x);
    
    
    // Initialize some elements needed in Bspline products creation ------------------------------------------------------
    ts = std::make_shared<Bspline>();
    ts->nt = getInput()->getNumSpatialDims();
    ts->Dp[0] = pIP->Dp[0];
    ts->Dp[1] = pIP->Dp[1];
    ts->E = pIP->E; // Orden de splines
    ts->c[0] = float(bound_box->at(0) + bound_box->at(2)) / 2; // Vertical
    ts->c[1] = float(bound_box->at(1) + bound_box->at(3)) / 2; // Horizontal
    ts->rp[0] = (ts->E + 1) * ts->Dp[0] / 2; // Radio de influencia
    ts->rp[1] = (ts->E + 1) * ts->Dp[1] / 2; // Radio de influencia

    for(uint l = 0; l < ts->nt; l++) { // dimensiones
	for(uint b = 0; b < 2; b++) { // min-max
	    ts->C[l][b] = floor((abs(ts->c[l] - bound_box->at(l + b * 2)) + ts->rp[l]) / ts->Dp[l]);
	}
    }
    
    
    
    // Empty XData creation -------------------------------------------------------------------------------------
    std::vector<std::vector<dimIndexType>*>* pNDArrayDims;
    std::vector<dimIndexType>* pNDArrayDynDims;
    
    uint dp0 = (uint)(2 * ceil(ts->rp[0]) + 1); // Di�metro de influencia del spline
    uint dp1 = (uint)(2 * ceil(ts->rp[1]) + 1); // Di�metro de influencia del spline
    uint Cp1 = (uint)(ts->C[1][0] + ts->C[1][1] + 1);
    uint Cp0 = (uint)(ts->C[0][0] + ts->C[0][1] + 1);
    

    ts->coef = std::make_shared<XData>(getApp(), nCOLS, 2, ts->nt, TYPEID_REAL);
    
    ts->coefg = std::make_shared<XData>(getApp(), Cp1, 2, ts->nt, TYPEID_REAL);
    
    puData = std::make_shared<XData>(getApp(), Cp0, Cp1, ts->nt, TYPEID_REAL);

    
    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>({new std::vector<dimIndexType>({nCOLS, ts->E + 1, 3, ts->nt})});
    BBAuxData = std::make_shared<XData>(getApp(), pNDArrayDims, TYPEID_REAL);
    
    
    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>({new std::vector<dimIndexType>({Cp1, dp0, 3, 2})});
    BBAux1Data = std::make_shared<XData>(getApp(), pNDArrayDims, TYPEID_REAL);
    
    
    // Meter en el mismo Data 4 NDArrays que contengan auxBB, auxBB1, auxBB2, auxBB11
    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>();
    pNDArrayDims->push_back(new std::vector<dimIndexType>({ts->E + 1, ts->E + 1, nCOLS, nROWS}));
    pNDArrayDims->push_back(new std::vector<dimIndexType>({ts->E + 1, ts->E + 1, nCOLS, nROWS, ts->nt}));
    pNDArrayDims->push_back(new std::vector<dimIndexType>({ts->E + 1, ts->E + 1, nCOLS, nROWS, ts->nt}));
    pNDArrayDims->push_back(new std::vector<dimIndexType>({ts->E + 1, ts->E + 1, nCOLS, nROWS}));
    pNDArrayDynDims = new std::vector <dimIndexType>({4});
    auxBBallData = std::make_shared<XData>(getApp(), pNDArrayDims, pNDArrayDynDims, TYPEID_REAL);
    
    
    // Meter en el mismo Data 4 NDArrays que contengan BB, BB1, BB2, BB11
    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>();
    pNDArrayDims->push_back(new std::vector<dimIndexType>({nCOLS, nROWS, nFRAMES, ts->nt, ts->E + 1, ts->E + 1}));
    pNDArrayDims->push_back(new std::vector<dimIndexType>({nCOLS, nROWS, nFRAMES, ts->nt, ts->E + 1, ts->E + 1, 2}));
    pNDArrayDims->push_back(new std::vector<dimIndexType>({nCOLS, nROWS, nFRAMES, ts->nt, ts->E + 1, ts->E + 1, 2}));
    pNDArrayDims->push_back(new std::vector<dimIndexType>({nCOLS, nROWS, nFRAMES, ts->nt, ts->E + 1, ts->E + 1}));
    pNDArrayDynDims = new std::vector <dimIndexType>({4});
    ts->BBall = std::make_shared<XData>(getApp(), pNDArrayDims, pNDArrayDynDims, TYPEID_REAL);
    
    
    // Meter en el mismo Data 3 NDArrays que contengan auxBB1g, auxBB2g, auxBB11g
    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>();
    pNDArrayDims->push_back(new std::vector<dimIndexType>({dp1, dp0, Cp1, Cp0, 2}));
    pNDArrayDims->push_back(new std::vector<dimIndexType>({dp1, dp0, Cp1, Cp0, 2}));
    pNDArrayDims->push_back(new std::vector<dimIndexType>({dp1, dp0, Cp1, Cp0}));
    pNDArrayDynDims = new std::vector <dimIndexType>({3});
    auxBBgallData = std::make_shared<XData>(getApp(), pNDArrayDims, pNDArrayDynDims, TYPEID_REAL);
    
    
    // Meter en el mismo Data 3 NDArrays que contengan aux1BB1g, aux1BB2g, aux1BB11g
    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>();
    pNDArrayDims->push_back(new std::vector<dimIndexType>({dp1, dp0, Cp1, Cp0, 2, nFRAMES, 2}));
    pNDArrayDims->push_back(new std::vector<dimIndexType>({dp1, dp0, Cp1, Cp0, 2, nFRAMES, 2}));
    pNDArrayDims->push_back(new std::vector<dimIndexType>({dp1, dp0, Cp1, Cp0, nFRAMES, 2}));
    pNDArrayDynDims = new std::vector <dimIndexType>({3});
    aux1BBgallData = std::make_shared<XData>(getApp(), pNDArrayDims, pNDArrayDynDims, TYPEID_REAL);
    
    
     // Meter en el mismo Data 4 NDArrays que contengan BBg, BB1g, BB2g, BB11g
    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>();
    pNDArrayDims->push_back(new std::vector<dimIndexType>({dp1, dp0, Cp1, Cp0}));
    pNDArrayDims->push_back(new std::vector<dimIndexType>({dp1, dp0, nFRAMES, Cp1, Cp0, 2, 2}));
    pNDArrayDims->push_back(new std::vector<dimIndexType>({dp1, dp0, nFRAMES, Cp1, Cp0, 2, 2}));
    pNDArrayDims->push_back(new std::vector<dimIndexType>({dp1, dp0, nFRAMES, Cp1, Cp0, 2}));
    pNDArrayDynDims = new std::vector <dimIndexType>({4});
    ts->BBgall = std::make_shared<XData>(getApp(), pNDArrayDims, pNDArrayDynDims, TYPEID_REAL); // only copies dimensions and allocs memory
    
    
    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>();
    pNDArrayDims->push_back(new std::vector<dimIndexType>({longr2, longr1, nFRAMES, ts->nt, ts->E + 1, ts->E + 1}));
    pNDArrayDims->push_back(new std::vector<dimIndexType>({longr1, longr2, nFRAMES, ts->nt, ts->E + 1, ts->E + 1})); // TAuxPermuted
    pNDArrayDims->push_back(new std::vector<dimIndexType>({longr1, longr2, nFRAMES, ts->nt, ts->E + 1, ts->E + 1})); // TAuxShift1
    pNDArrayDims->push_back(new std::vector<dimIndexType>({longr1, longr2, nFRAMES, ts->nt, ts->E + 1, ts->E + 1})); // TAuxShift2
    pNDArrayDynDims = new std::vector <dimIndexType>({4});
    TAuxData = std::make_shared<XData>(getApp(), pNDArrayDims, pNDArrayDynDims, TYPEID_REAL); // only copies dimensions and allocs memory
    
    
    // XData containig the metric - V
    VData = std::make_shared<XData>(getApp(), nCOLS, nROWS, TYPEID_REAL); // only copies dimensions and allocs memory
    
    
    // XData containing regularization terms
    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>();
    pNDArrayDims->push_back(new std::vector<dimIndexType>({nCOLS, nROWS, nFRAMES, ts->nt, 2})); // dtaux
    pNDArrayDims->push_back(new std::vector<dimIndexType>({nCOLS, nROWS, nFRAMES, ts->nt, 2})); // dtaux2
    pNDArrayDims->push_back(new std::vector<dimIndexType>({nCOLS, nROWS, nFRAMES, ts->nt})); // dtauxy
    pNDArrayDims->push_back(new std::vector<dimIndexType>({nCOLS, nROWS, nFRAMES, ts->nt})); // dtaut
    pNDArrayDims->push_back(new std::vector<dimIndexType>({nCOLS, nROWS, nFRAMES, ts->nt})); // dtaut2
    pNDArrayDynDims = new std::vector <dimIndexType>({5});
    dtauallData = std::make_shared<XData>(getApp(), pNDArrayDims, pNDArrayDynDims, TYPEID_REAL); // only copies dimensions and allocs memory
    
    
    // XData containing transformation gradient
    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>();
    pNDArrayDims->push_back(new std::vector<dimIndexType>({dp1, dp0, nFRAMES, Cp1, Cp0, ts->nt, 2})); // dthetax
    pNDArrayDims->push_back(new std::vector<dimIndexType>({dp1, dp0, nFRAMES, Cp1, Cp0, ts->nt, 2})); // dthetax2
    pNDArrayDims->push_back(new std::vector<dimIndexType>({dp1, dp0, nFRAMES, Cp1, Cp0, ts->nt})); // dthetaxy
    pNDArrayDims->push_back(new std::vector<dimIndexType>({dp1, dp0, nFRAMES, Cp1, Cp0, ts->nt})); // dthetat
    pNDArrayDims->push_back(new std::vector<dimIndexType>({dp1, dp0, nFRAMES, Cp1, Cp0, ts->nt})); // dthetat2
    pNDArrayDynDims = new std::vector <dimIndexType>({5});
    dthetaallData = std::make_shared<XData>(getApp(), pNDArrayDims, pNDArrayDynDims, TYPEID_REAL); // only copies dimensions and allocs memory
    
    
    // XData containing metric cost
    HData = std::make_shared<XData>(getApp(), getParametersGD().nmax + 1, TYPEID_REAL); // only copies dimensions and allocs memory
    
    
    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>();
    pNDArrayDims->push_back(new std::vector<dimIndexType>({nCOLS, nROWS, nFRAMES, 2})); // d1
    pNDArrayDims->push_back(new std::vector<dimIndexType>({nCOLS, nROWS, nFRAMES, 2})); // d2
    delete(pNDArrayDynDims); pNDArrayDynDims = new std::vector <dimIndexType>({2});
    dallData = std::make_shared<XData>(getApp(), pNDArrayDims, pNDArrayDynDims, TYPEID_REAL);
    
    
    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>({new std::vector<dimIndexType>({nCOLS, nROWS, nFRAMES, 2})});
    dxData = std::make_shared<XData>(getApp(), pNDArrayDims, TYPEID_REAL);

    
    dyData = std::make_shared<XData>(getApp(), nCOLS, nROWS, nFRAMES, TYPEID_REAL);
    
    
    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>({new std::vector<dimIndexType>({nCOLS, nROWS, nFRAMES, ts->nt})});
    xnData = std::make_shared<XData>(getApp(), pNDArrayDims, TYPEID_REAL);
    
    
    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>({new std::vector<dimIndexType>({dp1, dp0, nFRAMES, Cp1, Cp0, ts->nt})}); // aux 
    auxData = std::make_shared<XData>(getApp(), pNDArrayDims, TYPEID_REAL);
    

    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>({new std::vector<dimIndexType>({dp1, dp0, nFRAMES, Cp1, Cp0})}); // aux1
    aux1Data = std::make_shared<XData>(getApp(), pNDArrayDims, TYPEID_REAL);
    

    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>({new std::vector<dimIndexType>({dp1, dp0, Cp1, Cp0})}); // aux2
    aux2Data = std::make_shared<XData>(getApp(), pNDArrayDims, TYPEID_REAL); 

    
    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>({new std::vector<dimIndexType>({dp1, dp0, nFRAMES, Cp1, Cp0})});
    AuxData = std::make_shared<XData>(getApp(), pNDArrayDims, TYPEID_REAL);
    

    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>();
    pNDArrayDims->push_back(new std::vector<dimIndexType>({dp1, dp0, nFRAMES, Cp1, Cp0, ts->nt})); // dV
    pNDArrayDims->push_back(new std::vector<dimIndexType>({dp1, dp0, Cp1, Cp0, nFRAMES, ts->nt})); // permutedV
    pNDArrayDynDims = new std::vector <dimIndexType>({2});
    dVData = std::make_shared<XData>(getApp(), pNDArrayDims, pNDArrayDynDims, TYPEID_REAL);
    
    
    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>({new std::vector<dimIndexType>({dp1, dp0, nFRAMES, ts->nt})}); // Xaux
    XauxData = std::make_shared<XData>(getApp(), pNDArrayDims, TYPEID_REAL); 
    

    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>({new std::vector<dimIndexType>({dp1, dp0, nFRAMES, ts->nt})}); // cost1 
    cost1Data = std::make_shared<XData>(getApp(), pNDArrayDims, TYPEID_REAL); 
    

    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>({new std::vector<dimIndexType>({dp1, nFRAMES, ts->nt})}); // cost2
    cost2Data = std::make_shared<XData>(getApp(), pNDArrayDims, TYPEID_REAL); 
    

    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>({new std::vector<dimIndexType>({Cp0, Cp1, nFRAMES, ts->nt})});
    dHData = std::make_shared<XData>(getApp(), pNDArrayDims, TYPEID_REAL);
    

    pNDArrayDims = new std::vector<std::vector<dimIndexType>*>({new std::vector<dimIndexType>({Cp1, Cp0, nFRAMES, ts->nt})});
    DifData = std::make_shared<XData>(getApp(), pNDArrayDims, TYPEID_REAL); 
    
    
    
    // Initialize subprocesses
    pDataNormalization->init();
    
    pControlPoints->init();
    pCreateBB->init();
    pPermuteBB->init();
    pCreateBBg->init();
    pRepmatBBg->init();
    pPermuteBBg->init();

    pInitZero->init();
    pInitIT->init();
    pMetric->init();
    pGradient->init();
    pPermutedV->init();
    pUpdateTransformation->init();
    pCost6D->init();
    pGradientMetric->init();
    pGradientJointAux->init();
    pGradientWithSmoothTerms->init();
    pReduction7Dto6D->init();
    pReduction6Dto5D->init();
    pReduction5Dto4D->init();
    pAuxiliarMask->init();
    pAuxiliarCost->init();
    pCost6DReduction->init();
    pCostReduction->init();
    pDeformationAdjust->init();
    pInitdx->init();
    pPermuteTAux->init();
    pShiftTAux->init();
    pGradientRegularization->init();
    
    
    auto iTransAuxParms = std::make_shared<TransformationAux::InitParameters>(longr1, longr2, bound_box);
    pTransformationAux->setInitParameters(iTransAuxParms);
    pTransformationAux->init();
    
    auto iDeforParms = std::make_shared<Deformation::InitParameters>(longr1, longr2, bound_box);
    pDeformation->setInitParameters(iDeforParms);
    pDeformation->init();
    
    auto iGradIntParms = std::make_shared<GradientInterpolator::InitParameters>(longr1margin, longr2margin, margin, bound_box);
    pGradientInterpolator->setInitParameters(iGradIntParms);
    pGradientInterpolator->init();
    
    auto iRegParms = std::make_shared<Regularization::InitParameters>(longr1, longr2, bound_box);
    pRegularization->setInitParameters(iRegParms);
    pRegularization->init();
    
    auto iSumLambdaParms = std::make_shared<SumLambdaMetric::InitParameters>(longr1, longr2, bound_box);
    pSumLambdaMetric->setInitParameters(iSumLambdaParms);
    pSumLambdaMetric->init();

}

/**
 * @brief  Assign weights to smoothnes terms
 * @param[in] lambda vector containing weights to smoothness terms {first spatial, second spatial, first temporal, second temporal}
 */
void GroupwiseRegistration::setTerms(const std::vector<float> &lambda) {
    this->term.lambda = lambda;
}

/**
 * @brief Assign Gradient Descent algorithm parameters
 * @param[in] paramNmax Maximum number of iterations
 * @param[in] paramEt Transformation norm threshold
 * @param[in] paramEh Metric variation threshold
 */
void GroupwiseRegistration::setParametersGD(const unsigned int paramNmax, const float paramEt, const float paramEh) {
    this->param.nmax = paramNmax;
    this->param.et = paramEt;
    this->param.eh = paramEh;
}

/**
 * @brief Sums all lambda terms (four terms)
 * @return sum of all lambda terms
 */
float GroupwiseRegistration::sumLambda() {
    return std::accumulate(term.lambda.begin(), term.lambda.end(), 0);
}


/**
 * @brief Calculates the differences between inputs (norm and y metric) and recast them to adequeate them to output conditions
 * @param[out] Tdif norm difference of the transformation matrix
 * @param[out] Hdif metric variation
 * @param[out] Wn weight
 * @param[in] Dif transformation matrix (displacement)
 * @param[in] H values of the metric for each iteration
 * @param[in] iter current iteration
 * @param[in] flagW flag for adaptive W algorithm
 */
void GroupwiseRegistration::evolution(float* Tdif, float* Hdif, float* Wn, realType* Dif, realType* H, int iter, bool flagW) {
    // Calculus of the differences
    float sum_squares = 0.0f;
    const std::vector<dimIndexType> dataSize = *(DifData->getNDArray(0)->getDims());
    dimIndexType totalDim = 1;
    for(int n : dataSize)
	totalDim *= n;
    for(dimIndexType i = 0; i < totalDim; i++) {
	sum_squares += Dif[i] * Dif[i];
    }
    Tdif[iter - 1] = sqrt(sum_squares);     // Norm
    Hdif[iter - 1] = H[iter - 1] - H[iter]; // Metric
    if(flagW) {                             // Adaptation of Wn, flagW
	if(Hdif[iter - 1] > 0) {
	    (*Wn) = (*Wn) * 1.2;
	}
	else {
	    (*Wn) = (*Wn) / 2;
	}
    }
}

/**
 * @brief Draws a circle of a given radius over the ROI in the center of the image
 * @param[out] X matrix representing the circular mask
 * @param[out] bound_box bounding box of X (ROI)
 * @param[in] radius radius of the circle
 * @return total area of the ROI (Region Of Interest)
 */
int GroupwiseRegistration::setMask(std::vector<dimIndexType>* X, std::vector<dimIndexType>* bound_box, const int radius) {
    GROUPWISEREGISTRATION_CERR("\n ---> Setting mask with radius " << std::to_string(radius) << "...\n" << std::endl);
    uint nROWS = getInput()->getSpatialDimSize(ROWS, 0);
    uint nCOLS = getInput()->getSpatialDimSize(COLUMNS, 0);
    int radius2 = pow(radius, 2); // Precomputing the squared radius
    // Center coordinates
    float center[2] = {float(nROWS + 1) / 2, float(nCOLS + 1) / 2};
    GROUPWISEREGISTRATION_CERR("\tCenter: [" << std::to_string(center[0]) << "," << std::to_string(center[1]) << "]" << std::endl);
    // Mesh of the distances to the center of each pixel
    float C[nCOLS][nROWS];
    float D[nCOLS][nROWS];
    int areaX = 0;
    uint min_fil1 = nCOLS;
    uint max_fil1 = 0;
    uint min_col1 = nROWS;
    uint max_col1 = 0;
    for(dimIndexType i = 0; i < nCOLS; i++)	{ // Rows (height)
	for(dimIndexType j = 0; j < nROWS; j++)	{ // Columns (width)
	    C[i][j] = pow(1 - center[0] + j, 2); // Horizontal axis
	    D[i][j] = pow(1 - center[1] + i, 2); // Vertical axis
	    if((C[i][j] + D[i][j]) <= radius2) {
		X->at(i + j * nCOLS) = 1;
		if(i < min_fil1) // Get minimum row index
		    min_fil1 = i;
		if(j < min_col1) // Get minimum column index
		    min_col1 = j;
		if(i > max_fil1) // Get maximum row index
		    max_fil1 = i;
		if(j > max_col1) // Get maximum column index
		    max_col1 = j;
		areaX++; // Area of the ROI
	    }
	}
    }
    // Bound_box: limits of the ROI
    bound_box->at(0) = min_fil1 + 1;
    bound_box->at(1) = min_col1 + 1;
    bound_box->at(2) = max_fil1 + 1;
    bound_box->at(3) = max_col1 + 1;
    return areaX;
}

/**
 * @brief Check the condition to exit the optimization loop
 * @param[in] p1 Tdif in current iteration
 * @param[in] p3 Hdif in current iteration
 * @param[in] p2 threshold to compare Tdif
 * @param[in] p4 threshold to compare Hdif
 * @param[in] p5 max number of iterations in optimizer
 * @param[in] p6 current iteration in optimizer
 * @return True if the output condition is satisfied, false if not
 */
bool GroupwiseRegistration::stopCondition(const float p1, const float p3, const float p2, const float p4, const int p5, const int p6) {
    bool stop = false;
    if((p5 == p6) || ((p1 < p2) && (p3 < p4)))
	stop = true;
    return stop;
}



/**
 * @brief Launch a non rigid 2D groupwise registration based on B-splines with motion estimation
 * @param pProfileParameters->enable flag to enable profiling
 */
void GroupwiseRegistration::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);
    
    infoItems.addInfoItem("Title", "GroupwiseRegistration info");
    startProfiling();

    GROUPWISEREGISTRATION_CERR("\n\n|=============================================================================|" << std::endl);
    GROUPWISEREGISTRATION_CERR("|\tGROUPWISE REGISTRATION" << std::endl);
    GROUPWISEREGISTRATION_CERR("|=============================================================================|\n" << std::endl);
    

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Preparing the data                                                                    //
    // Images are complex. To run the code in a proper way we need to compute the modulus,   //
    // in a range between 0 and a maximum we define                                          //
    ///////////////////////////////////////////////////////////////////////////////////////////
    // 1. We need to know the max value over all images                                      //
    ///////////////////////////////////////////////////////////////////////////////////////////

    // We need the maximum, in modulus, of original images
    getInput()->device2Host();
    std::vector<float> maxImages(getInput()->getNumNDArrays());
    dimIndexType width, height, depth;
    for(uint i = 0; i < getInput()->getNumNDArrays(); i++) {
	width = NDARRAYWIDTH(getInput()->getNDArray(i));
	height = NDARRAYHEIGHT(getInput()->getNDArray(i));
	depth = NDARRAYDEPTH(getInput()->getNDArray(i));
	if(depth == 0)
	    depth = 1;

	std::vector<float> eachImageAbs(width * height);
	complexType* pComplexArray;
	pComplexArray = (complexType*) getInput()->getHostBuffer(i);
	for(dimIndexType index1D = 0;  index1D < height * width * depth; index1D++) {
	    eachImageAbs[index1D] = std::fabs(pComplexArray[index1D]);
	}
	maxImages[i] = *max_element(eachImageAbs.begin(), eachImageAbs.end());
    }
    float oldmax = *max_element(maxImages.begin(), maxImages.end());
    float newmax = 500;

    ///////////////////////////////////////////////////////////////////////////////////////////
    // 2. Now we know which is the maximum, in module, overall images, and have to           //
    //    convert images to that range                                                       //
    ///////////////////////////////////////////////////////////////////////////////////////////
    GROUPWISEREGISTRATION_CERR("\n ---> Normalizing data from 0 to " << std::to_string(newmax) << "... ");
    pDataNormalization->setInput(getInput());
    pDataNormalization->setOutput(normInputData);
    auto parDataNormalization = std::make_shared<DataNormalization::LaunchParameters>(oldmax, newmax);
    pDataNormalization->setLaunchParameters(parDataNormalization);
    pDataNormalization->launch();
    

    // CREABSPLINE execution -------------------------------------------------------------
    if(pLP->currentMotionIter == 0) {
	creabspline();
    }

    // OPTIMIZATOR execution -------------------------------------------------------------
    optimizer(pLP->TData, pLP->argsMC);


    GROUPWISEREGISTRATION_CERR("\n|=============================================================================|" << std::endl);
    GROUPWISEREGISTRATION_CERR("|\tDONE" << std::endl);
    GROUPWISEREGISTRATION_CERR("|=============================================================================|\n" << std::endl);
    stopProfiling();
}
}
#undef GROUPWISEREGISTRATION_DEBUG
