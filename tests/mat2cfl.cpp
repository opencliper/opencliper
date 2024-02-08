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
//#define MAT2CFL_DEBUG

#if !defined NDEBUG && defined MAT2CFL_DEBUG
    #define MAT2CFL_CERR(x) CERR(x)
#else
    #define MAT2CFL_CERR(x)
    #undef MAT2CFL_DEBUG
#endif


using namespace OpenCLIPER;
int main(int argc, char* argv[]) {
    try {
		CLapp::PlatformTraits platformTraits;
		CLapp::DeviceTraits deviceTraits;
		PerfTestConfResult* pPerfTest = new PerfTestConfResult(argc, argv, " <inputFileName>*");
		auto pConfigTraits = std::dynamic_pointer_cast<PerfTestConfResult::ConfigTraits>(pPerfTest->getConfigTraits());
		std::string fileName;
		if (pConfigTraits->nonOptionArgs.size() == 0) {
			std::cerr << "Missing file name(s)" << std::endl;
			exit(-1);
		}
		deviceTraits = pConfigTraits->deviceTraits;
		platformTraits = pConfigTraits->platformTraits;
		// CL_QUEUE_PROFILING_ENABLE slows down the queue. Beware!
		// deviceTraits.queueProperties = cl::QueueProperties(CL_QUEUE_PROFILING_ENABLE);
		auto pCLapp = CLapp::create(platformTraits, deviceTraits);

		std::chrono::high_resolution_clock::time_point beginLoadTime;
		std::chrono::high_resolution_clock::time_point endLoadTime;
		TIME_DIFF_TYPE elapsedTime;
		std::shared_ptr<KData> pInputKData;
		for (unsigned int kDataIndex = 0; kDataIndex < pConfigTraits->nonOptionArgs.size(); kDataIndex++) {
			// Load input data from Matlab file
			fileName = pConfigTraits->nonOptionArgs.at(kDataIndex);
			// Start measuring loading time now
			std::cout << "Loading file: " << fileName << "... " << std::flush;
			beginLoadTime = std::chrono::high_resolution_clock::now();
			pInputKData = std::make_shared<KData>(pCLapp, fileName);
			// Load time measurement ends now
			endLoadTime = std::chrono::high_resolution_clock::now();
			elapsedTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(endLoadTime - beginLoadTime).count()) / 1e9;
			std::cout << "Done." << std::endl;
			std::ostringstream ostream2;
			ostream2 << std::fixed << std::setprecision(PROFILINGTIMESPRECISION) << elapsedTime;
			std::cout << " --> Time elapsed in loading file: " << ostream2.str() << "\n" << std::endl;
			if (LPISupport::Utils::extensionname(fileName) == "mat") {
				std::cout << "Saving corresponding CFL files... " << std::flush;
				beginLoadTime = std::chrono::high_resolution_clock::now();
				pInputKData->saveCFLData(LPISupport::Utils::basename(fileName));
				endLoadTime = std::chrono::high_resolution_clock::now();
			} else {
				std::cout << "Saving corresponding Matlab file... " << std::flush;
				beginLoadTime = std::chrono::high_resolution_clock::now();
				pInputKData->matlabSave(LPISupport::Utils::basename(fileName) + ".mat");
				endLoadTime = std::chrono::high_resolution_clock::now();
			}

			elapsedTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(endLoadTime - beginLoadTime).count()) / 1e9;
			std::cout << "Done. (" << elapsedTime << " s)" << std::endl;
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

#undef MAT2CFL_DEBUG
