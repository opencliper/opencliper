/* Copyright (C) 2018 Federico Simmross Wattenberg,
 *                    Manuel Rodr�guez Cayetano,
 *                    Javier Royuela del Val,
 *                    Elena Mart�n Gonz�lez,
 *                    Elisa Moya S�ez,
 *                    Marcos Mart�n Fern�ndez and
 *                    Carlos Alberola L�pez
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
 *  E.T.S.I. Telecomunicaci�n
 *  Universidad de Valladolid
 *  Paseo de Bel�n 15
 *  47011 Valladolid, Spain.
 *  fedsim@tel.uva.es
 */
/*
 * FFT.hpp
 *
 *  Created on: 23 de nov. de 2016
 *      Author: fedsim
 */

#ifndef FFT_HPP
#define FFT_HPP

#include <OpenCLIPER/Process.hpp>
#include <OpenCLIPER/processes/ScalarMultiply.hpp>
#include <OpenCLIPER/Data.hpp>
#include <OpenCLIPER/SamplingMasksData.hpp>
#include <clFFT.h>

#ifdef HAVE_ROCFFT
    #include <rocfft.h>
#endif

// CLFFT_SINGLE_FAST and CLFFT_DOUBLE_FAST are apparently not
// implemented in clFFT right now
#ifdef DOUBLE_PREC
    #define OPENCLIPER_CLFFT_PRECISION CLFFT_DOUBLE
    #define OPENCLIPER_ROCFFT_PRECISION rocfft_precision_double
#else
    #define OPENCLIPER_CLFFT_PRECISION CLFFT_SINGLE
    #define OPENCLIPER_ROCFFT_PRECISION rocfft_precision_single
#endif


namespace OpenCLIPER {
/**
 * @brief Process class for the FFT operation on a image
 *
 */
class FFT: public Process {
    public:
        ~FFT();

	/// Enumerated type with direction of FFT
	enum Direction {
	    /// Forward direction
	    FORWARD = CLFFT_FORWARD,
	    /// Backward direction
	    BACKWARD = CLFFT_BACKWARD
	};

	/// Parameters related to process initialization
	struct InitParameters: Process::InitParameters {
	    /// Do the FFT along this dimension only (-1 means all dimensions)
	    int dim;

	    /// For the special case of the first spatial dimension (i.e. rows), do the FFT at these lines only (nullptr means all rows)
	    std::shared_ptr<SamplingMasksData> samplingMask;

	    /// constructor
	    explicit InitParameters(int d = -1, std::shared_ptr<SamplingMasksData> s = nullptr): dim(d), samplingMask(s) {}
	};

	/// Parameters related to kernel execution
	struct LaunchParameters: Process::LaunchParameters {
	    /// direction of FFT
	    Direction dir;

	    /// constructor
	    explicit LaunchParameters(Direction d = Direction::FORWARD): dir(d) {}
	};

	// Methods
	void init();
	void launch();

    private:
	using Process::Process;

	/// handle to FFT plan (FFT configuration)
	clfftPlanHandle clPlanHandle;

	/// number of batches needed to cover the whole volume (varies with specified dim and samplingMask values)
	size_t nBatches;

	/// offsets to start position of each batch within the CL input/output buffers (varies with specified dim and samplingMask values)
	std::vector<size_t> batchOffsets;

        // Work buffer
        std::shared_ptr<Data> clWorkBuffer = nullptr;

#ifdef HAVE_ROCFFT
	rocfft_plan     rocPlanHandleFW;
	rocfft_plan     rocPlanHandleBW;

	rocfft_plan_description rocfftPlanDescription;

	void*           rocWorkBufferFW = nullptr;
	size_t          rocWorkBufferBytesFW = 0;
	void*           rocWorkBufferBW = nullptr;
	size_t          rocWorkBufferBytesBW = 0;

	rocfft_execution_info rocExecInfoFW;
	rocfft_execution_info rocExecInfoBW;

	std::shared_ptr<ScalarMultiply> scalarMultiply;
	std::shared_ptr<ScalarMultiply::LaunchParameters> scalarMultiplyLP;
#endif
        //Quitar
        std::shared_ptr<ScalarMultiply> scalarMultiply;
	std::shared_ptr<ScalarMultiply::LaunchParameters> scalarMultiplyLP;

};

} // namespace OpenCLIPER

#endif // FFT_HPP
