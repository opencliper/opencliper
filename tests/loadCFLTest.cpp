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

// Uncomment to show specific debug messages
//#define LOADCFLTEST_DEBUG

#if !defined NDEBUG && defined LOADCFLTEST_DEBUG
    #define LOADCFLTEST_CERR(x) CERR(x)
#else
    #define LOADCFLTEST_CERR(x)
    #undef LOADCFLTEST_DEBUG
#endif


using namespace OpenCLIPER;
int main(int argc, char* argv[]) {
    std::shared_ptr<LPISupport::SampleCollection> pSamples = std::make_shared<LPISupport::SampleCollection>("execution time");
    try {
	CLapp::PlatformTraits platformTraits;
	CLapp::DeviceTraits deviceTraits;
	PerfTestConfResult* pPerfTest = new PerfTestConfResult(argc, argv, " <inputFileName>");
	auto pConfigTraits = std::dynamic_pointer_cast<PerfTestConfResult::ConfigTraits>(pPerfTest->getConfigTraits());
	unsigned int numberOfIterations = pConfigTraits->repetitions;
	std::cerr << "Number of iterations: " << numberOfIterations << "\n";
	std::string fileName;
	if (pConfigTraits->nonOptionArgs.size() != 0) {
		// Load input data from Matlab file
		fileName = pConfigTraits->nonOptionArgs.at(0);
	} else {
		std::cerr << "Missing file name" << std::endl;
		exit(-1);
	}
	deviceTraits = pConfigTraits->deviceTraits;
	platformTraits = pConfigTraits->platformTraits;
	// CL_QUEUE_PROFILING_ENABLE slows down the queue. Beware!
	// deviceTraits.queueProperties = cl::QueueProperties(CL_QUEUE_PROFILING_ENABLE);
	auto pCLapp = CLapp::create(platformTraits, deviceTraits);

	std::chrono::high_resolution_clock::time_point beginLoadExecTime;
	std::chrono::high_resolution_clock::time_point endLoadExecTime;
	TIME_DIFF_TYPE elapsedTimeLoad;
	std::shared_ptr<KData> pInputKData;
	for(unsigned int iterNum = 0; iterNum < numberOfIterations; iterNum++) {
		// Start measuring execution time now
		beginLoadExecTime = std::chrono::high_resolution_clock::now();
		pInputKData = std::make_shared<KData>(pCLapp, fileName);
		// Execution time measurement ends now
		endLoadExecTime = std::chrono::high_resolution_clock::now();
		elapsedTimeLoad = (std::chrono::duration_cast<std::chrono::nanoseconds>(endLoadExecTime - beginLoadExecTime).count()) / 1e9;

		std::ostringstream ostream2;
		ostream2 << std::fixed << std::setprecision(PROFILINGTIMESPRECISION) << elapsedTimeLoad;
		std::cout << "iter_number: " << iterNum << " --> Time elapsed in loading file: " << ostream2.str() << "\n" << std::endl;
        pSamples->appendSample(elapsedTimeLoad);
	}
	pInputKData->show();
	pInputKData->getSensitivityMapsData()->show();
	pInputKData->getSamplingMasksData()->show();
	pInputKData->saveCFLHeader("prueba.hdr");
	auto pIn = std::make_shared<XData>(pCLapp, DATA_DIR "/Cameraman.tif", TYPEID_COMPLEX);
	pIn->saveRawData("Cameraman.cfl");
	auto pInputKData2 = std::make_shared<KData>(pCLapp, "Cameraman.cfl");
	pInputKData2->matlabSave("Cameraman.mat");
	pInputKData2->show();
	pPerfTest->buildTestInfo(pSamples, &pCLapp);
	pPerfTest->saveOrPrint();
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

#undef LOADCFLTEST_DEBUG
