/* Copyright (C) 2018 Federico Simmross Wattenberg,
 *                    Manuel RodrÃ­guez Cayetano,
 *                    Javier Royuela del Val,
 *                    Elena MartÃ­n GonzÃ¡lez,
 *                    Elisa Moya SÃ¡ez,
 *                    Marcos MartÃ­n FernÃ¡ndez and
 *                    Carlos Alberola LÃ³pez
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
 *  E.T.S.I. TelecomunicaciÃ³n
 *  Universidad de Valladolid
 *  Paseo de BelÃ©n 15
 *  47011 Valladolid, Spain.
 *  fedsim@tel.uva.es
 */
#include <OpenCLIPER/defs.hpp>
#include <OpenCLIPER/KData.hpp>
#include <OpenCLIPER/XData.hpp>
#include <OpenCLIPER/processes/nesta/NestaUp.hpp>
#include <OpenCLIPER/buildconfig.hpp>
#include <OpenCLIPER/PerfTestConfResult.hpp>
#include <LPISupport/Timer.hpp>

// Uncomment to show specific debug messages
//#define MRIRECON_DEBUG
#define DEBUG_TIMES
#define ASYNCLOAD

#if !defined NDEBUG && defined MRIRECON_DEBUG
    #define MRIRECON_CERR(x) CERR(x)
#else
    #define MRIRECON_CERR(x)
    #undef MRIRECON_DEBUG
#endif


using namespace OpenCLIPER;
int main(int argc, char* argv[]) {
#ifdef DEBUG_TIMES
    std::shared_ptr<LPISupport::SampleCollection> pSamples = std::make_shared<LPISupport::SampleCollection>("execution time");
    std::shared_ptr<LPISupport::SampleCollection> pInitializationTimeSamples = std::make_shared<LPISupport::SampleCollection>("initialization time");
    std::shared_ptr<LPISupport::SampleCollection> pMotionCompensSamples = std::make_shared<LPISupport::SampleCollection>("motion compensation time");
#endif

    try {
	CLapp::PlatformTraits platformTraits;
	CLapp::DeviceTraits deviceTraits;
	PerfTestConfResult* pPerfTest = new PerfTestConfResult(argc, argv, " [<inputFileName>]*");
	auto pConfigTraits = std::dynamic_pointer_cast<PerfTestConfResult::ConfigTraits>(pPerfTest->getConfigTraits());
	unsigned int numberOfIterations = pConfigTraits->repetitions;
	bool showTimes = pConfigTraits->showTimes;

	CERR("Number of iterations: " << numberOfIterations << "\n");

	deviceTraits = pConfigTraits->deviceTraits;
	platformTraits = pConfigTraits->platformTraits;
	// CL_QUEUE_PROFILING_ENABLE slows down the queue. Beware!
	// deviceTraits.queueProperties = cl::QueueProperties(CL_QUEUE_PROFILING_ENABLE);

	LPISupport::Timer initDeviceTimer;
	auto pCLapp = CLapp::create(platformTraits, deviceTraits);
	if(showTimes) std::cerr << "Initialized computing device in " << initDeviceTimer.get() << " sec" << '\n';

		std::vector<std::string>* pFilenamesVector = new std::vector<std::string>();
		if (pConfigTraits->nonOptionArgs.size() != 0) {
			// Load input data from Matlab file
			for (unsigned int kDataIndex = 0; kDataIndex < pConfigTraits->nonOptionArgs.size(); kDataIndex++) {
				pFilenamesVector->push_back(pConfigTraits->nonOptionArgs.at(kDataIndex));
			}
		} else {
			CERR("No filename given. Using default\n");
// 			pFilenamesVector->push_back(DATA_DIR "/data_160x160_coils23_frames20_single.mat");
			pFilenamesVector->push_back(DATA_DIR "/phantomSA_2mm_BH_20phases_8coils_AF4_cube.mat");
		}

		// Create NestaUp process
		auto nestaUpProcess = Process::create<NestaUp>(pCLapp);

	std::shared_ptr<Data> pInputKData;
	std::shared_ptr<Data> pInputKData2;
	std::shared_ptr<XData> pOutputXData;
	std::string prevFilename;
	std::string filename;

	// Parameters for NESTA execution
	// -------------------------------------------------------------------------------------------------

	//int verbose = 10;
	int verbose = 1;
	uint maxIntIter = 4;
	int maxIter = 100;
	float tolVar = 3E-4;
	uint stoptest = 1;
	uint miniter = 7;
	float mu_f = 3E-4;
	float La = 1;
	float lambda_i = 1E-2;

	std::cerr << "tolVar: " << tolVar << std::endl;

	// Parameters for GROUPWISE REGISTRATION execution
	// -------------------------------------------------------------------------------------------------
	uint MOTION_ITERS = pConfigTraits->numOfMotionCompensIters; 	// Number of iterations to perform GroupwiseRegistration + Nesta
	CERR("MOTION_ITERS: " << MOTION_ITERS << std::endl);
	uint nmax = 100; 	// Maximum number of iterations
	float et = 0.01f; 	// Transformation norm threshold
	float eh = 0.005f; 	// Metric variation threshold
	std::vector<float> lambda = {0.f, 0.005f, 0.f, 0.5f}; // Smoothnes/Regularization weights (1st spatial, 2nd spatial, 1st temp, 2nd temp)
	int radius = 40; 	// Radius of the circular ROI (0 if 'wholebox')
	int E = 3; 		// E: spline order
	int Dp[2] = {4, 4}; 	// Dp: point density
	float W = 1.f; 		// Weight for each point displacement
	bool flagW = true; 	// Flag for adaptative set of W


	LPISupport::Timer totalEnd2EndTimer;
        float totalMcElapsed = 0;
	LPISupport::Timer loadTimer;
	for(unsigned int iterNum = 0; iterNum < numberOfIterations; iterNum++) {
	    for (unsigned int kDataIndex = 0; kDataIndex < pFilenamesVector->size(); kDataIndex++) {
		// Load input data from Matlab file
		//std::string filename = DATA_DIR + pFilenamesVector->at(kDataIndex);
		//std::string filename = "/datos/BD/MRI/Cardio/Raw_Data/20171027_UVa_Cine_Santi/generated/171116_halfScan_zeroFill/" + pFilenamesVector->at(kDataIndex);
		prevFilename = filename;
		filename = pFilenamesVector->at(kDataIndex);
		if(filename != prevFilename) {
		    std::cerr << "Loading: " << filename << "... " << std::flush;
		    loadTimer.start();

#ifdef ASYNCLOAD
		    if (kDataIndex == 0) {
			if (pFilenamesVector->size() != 1) {
			    pInputKData2 = std::make_shared<KData>(pCLapp, pFilenamesVector->at(kDataIndex + 1), true);
			}
			pInputKData = std::make_shared<KData>(pCLapp, pFilenamesVector->at(kDataIndex), false);
		    } else if (kDataIndex == (pFilenamesVector->size() - 1)) {
			pInputKData = pInputKData2;
			pInputKData2 = nullptr;
		    } else {
			pInputKData = pInputKData2;
			pInputKData2 = std::make_shared<KData>(pCLapp, pFilenamesVector->at(kDataIndex + 1), true);
		    }
#else
					pInputKData = std::make_shared<KData>(pCLapp, filename);
#endif

		    if(showTimes) std::cerr << "Done: " << loadTimer.get() << " s" << '\n';
		    else std::cerr << '\n';
		} else
		    std::cerr << "Re-using: " << filename << std::endl;



		// Create processes and load needed kernels
		LPISupport::Timer initProcessesTimer;
                initProcessesTimer.start();
#ifdef MRIRECON_DEBUG
		auto MCProcess = Process::create<MotionCompensation>(pCLapp);
		MCProcess->init();
#endif
				pCLapp->loadKernels();
				pInputKData->waitLoadEnd();

				// Create output with suitable size
				pOutputXData = std::make_shared<XData>(pCLapp, std::dynamic_pointer_cast<KData>(pInputKData));


#ifdef MRIRECON_DEBUG
		if (pConfigTraits->showImagesOrVideos) {
		    std::dynamic_pointer_cast<KData>(pInputKData)->show();
		    std::dynamic_pointer_cast<KData>(pInputKData)->getSensitivityMapsData()->show();
		    std::dynamic_pointer_cast<KData>(pInputKData)->getSamplingMasksData()->show();
		}
#endif

				// Set input/output and initialize processes
				if (kDataIndex==0){
					nestaUpProcess->setInput(pInputKData);
					nestaUpProcess->setOutput(pOutputXData);
					nestaUpProcess->init();
				}
				else{
					nestaUpProcess->setInput(pInputKData);
					nestaUpProcess->setOutput(pOutputXData);
				}

				// Reuse process GroupwiseRegistration
				std::shared_ptr<Data> TData;

				std::shared_ptr<Process> GWRegistrationProcess = nullptr;
				std::shared_ptr<GroupwiseRegistration::LaunchParameters> paramsGW;
				if(MOTION_ITERS != 0) {
					GWRegistrationProcess = Process::create<GroupwiseRegistration>(pCLapp);
					GWRegistrationProcess->setInput(pOutputXData); // Only needs input
					auto initParms = std::make_shared<GroupwiseRegistration::InitParameters>(W, flagW, radius, E, Dp, nmax, et, eh, lambda);
					GWRegistrationProcess->setInitParameters(initParms);
					GWRegistrationProcess->init();
				}

		if(showTimes) {
		    auto initProcessesElapsed = initProcessesTimer.get();
		    std::cerr << "Initialized processes in " << initProcessesElapsed << " s" << std::endl;
		    pInitializationTimeSamples->appendSample(initProcessesElapsed);
		}


		///////////////////////////////////////////////////////////////////////////////////////////
		// NESTA - First reconstruction                                                          //
		///////////////////////////////////////////////////////////////////////////////////////////

		// Start measuring total reconstruction time now
		LPISupport::Timer reconTimer;
		reconTimer.start();

		auto paramsNestaUp = std::make_shared<NestaUp::LaunchParameters>(lambda_i, mu_f, La, maxIntIter, tolVar, verbose, maxIter, stoptest, miniter, nullptr, 
										 pConfigTraits->showImagesOrVideos);
		nestaUpProcess->setLaunchParameters(paramsNestaUp);
		nestaUpProcess->launch();


#ifdef MRIRECON_DEBUG
		pOutputXData->matlabSave(DEBUG_OUTPUT_DIR "/NESTA_NOMOTION_RECON.MAT");
#endif

		// Start measuring motion compensation time now
		LPISupport::Timer mcTimer;
                totalMcElapsed = 0;
                mcTimer.start();
		for(unsigned int i = 0; i < MOTION_ITERS; i++) {

		    ///////////////////////////////////////////////////////////////////////////////////////////
		    // Groupwise Registration to estimate cardiac motion                                     //
		    ///////////////////////////////////////////////////////////////////////////////////////////
		    ArgumentsMotionCompensation* argsMC = new ArgumentsMotionCompensation();
		    paramsGW = std::make_shared<GroupwiseRegistration::LaunchParameters>(argsMC, TData, i);
		    GWRegistrationProcess->setLaunchParameters(paramsGW);
		    GWRegistrationProcess->launch();
		    TData = argsMC->TData; // Reuse T data obtained in current iteration as initialization for next iteration

#ifdef MRIRECON_DEBUG
		    std::shared_ptr<XData> pOutputRegData = std::make_shared<XData>(pCLapp, pOutputXData, 0);
		    MCProcess->setLaunchParameters(std::make_shared<MotionCompensation::LaunchParameters>(argsMC));
		    MCProcess->setInput(pOutputXData);
		    MCProcess->setOutput(pOutputRegData);
		    MCProcess->launch();
		    //MCProcess.reset(nullptr);
		    //pOutputRegData->device2Host(SyncSource::BUFFER_ONLY);
		    pOutputRegData->matlabSave(DEBUG_OUTPUT_DIR + std::string("/registration_") + std::to_string(i + 1) + std::string("recon.mat"));
		    pOutputRegData = nullptr;
#endif

		    ///////////////////////////////////////////////////////////////////////////////////////////
		    // NESTA - Reconstruction with motion compensation                                       //
		    ///////////////////////////////////////////////////////////////////////////////////////////
		    paramsNestaUp = std::make_shared<NestaUp::LaunchParameters>(lambda_i, mu_f, La, maxIntIter, tolVar, verbose, maxIter, stoptest, miniter, argsMC);
		    
		    nestaUpProcess->setLaunchParameters(paramsNestaUp);
		    nestaUpProcess->launch();
		    delete argsMC;

#ifdef MRIRECON_DEBUG
		    //pOutputXData->device2Host(SyncSource::BUFFER_ONLY); // automatically called by matlabSave
		    pOutputXData->matlabSave(DEBUG_OUTPUT_DIR + std::string("/NESTA_motion") + std::to_string(i + 1) + std::string("recon.mat"));
#endif
		}


		// Execution time measurement ends now
		auto reconElapsed = reconTimer.get();
		auto mcElapsed = mcTimer.get();

                if(showTimes) {
                    std::cerr << "iter_number: " << iterNum << " --> End to end time [slice " << std::to_string(kDataIndex) << "]: "
                              << std::fixed << std::setprecision(PROFILINGTIMESPRECISION) << reconElapsed << " s\n";
                }   
		totalMcElapsed += mcElapsed;

                if (iterNum == 0) { // only save reconstructed data during first iteration
		    // Save reconstructed data
		    std::string saveFileName = "MRIReconEnd2EndResult_slice_" + std::to_string(kDataIndex);
                    //std::string saveFileName = "MRIReconEnd2EndResult_slice_" + std::to_string(kDataIndex) + ".mat";
		    std::cerr<<"Saving: " << saveFileName << "... " << std::flush;
		    LPISupport::Timer saveTimer;
		    //pOutputXData->saveCFLData(saveFileName, false);
		    pOutputXData->saveCFLData(saveFileName, true);
                    //pOutputXData->matlabSave(saveFileName + ".mat");
		    if(showTimes) std::cerr << " Done: " << saveTimer.get() << " s\n";
                    else std::cerr << '\n';
		}
		// Show reconstructed data only if requested and during last iteration
		if ((iterNum ==  (numberOfIterations-1)) && (pConfigTraits->showImagesOrVideos)) {
			pOutputXData->show();
		}
	    }

            // Print complete profiling information
            auto totalReconElapsed = totalEnd2EndTimer.get();
            if(showTimes) {
                std::cerr << "iter_number: " << iterNum << " --> End to end time [all slices]: "
                        << std::fixed << std::setprecision(PROFILINGTIMESPRECISION) << totalReconElapsed << " s\n";
                pSamples->appendSample(totalReconElapsed);
                pMotionCompensSamples->appendSample(totalMcElapsed);
            }

        }

        if(showTimes) {
            pPerfTest->buildTestInfo(pSamples, &pCLapp);
            pPerfTest->buildTestInfo(pInitializationTimeSamples, &pCLapp);
            pPerfTest->buildTestInfo(pMotionCompensSamples, &pCLapp);
            pPerfTest->saveOrPrint();
        }
    }
    catch(cl::BuildError& e) {
	CLapp::dumpBuildError(e);
    }
    catch(CLError& e) {
	std::cerr << CLapp::getOpenCLErrorInfoStr(e, std::string(argv[0]));
    }
    catch(std::exception& e) {
	LPISupport::Utils::showExceptionInfo(e, argv[0]);
    }
}
#undef MRIRECON_DEBUG
