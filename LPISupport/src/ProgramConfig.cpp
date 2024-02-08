#include <LPISupport/ProgramConfig.hpp>

// Uncomment to show class-specific debug messages
//#define PROGRAMCONFIG_DEBUG

#define PROGRAMCONFIG_DEBUG

#if !defined NDEBUG && defined PROGRAMCONFIG_DEBUG
    #define PROGRAMCONFIG_CERR(x) CERR(x)
#else
    #define PROGRAMCONFIG_CERR(x)
    #undef PROGRAMCONFIG_DEBUG
#endif

// special file containing this process' executable name
#define PROC_SELF_EXE "/proc/self/exe"

#include <system_error>
#include <limits.h>
#include <LPISupport/Utils.hpp>

namespace LPISupport {

ProgramConfig::ProgramConfig() {

}

/**
*
* @param argc number of program run arguments (including program name)
* @param argv array of text strings with the names of the program run arguments
*/
ProgramConfig::ProgramConfig(int argc, char* argv[], std::string extraSummary) {
    pConfigTraits = std::make_shared<ConfigTraits>();
    init(argc, argv, extraSummary);
}

void ProgramConfig::init(int argc, char* argv[], std::string extraSummary) {
    pConfigTraits->usageSummary += extraSummary;
    readExecArgs(argc, argv);
    checkRequiredArgsPresent();
    pConfigTraits->configure();
}

/**
 * @brief Destructor for class (empty)
 *
 */
ProgramConfig::~ProgramConfig() {

}

/**
* @brief Method for setting usageOptions, shortArgs and shortRequiredArgs class fields
* @param optionName letter for the short option of program
* @param optionParameterName name of the option parameter (emtpy if there is no parameter)
* @param explanation text explaining option and its parameter (if any)
* @param mandatory true value if option is mandatory, false if optional
*/
void ProgramConfig::ConfigTraits::addSupportedShortOption(char optionName, std::string optionParameterName, std::string explanation, bool mandatory) {
    shortArgs.append(std::string(1, optionName));
    usageOptions.append("-" + std::string(1, optionName) + "\t");
    if(!optionParameterName.empty()) {
	shortArgs.append(":");
	usageOptions.append("<" + optionParameterName + ">\t");
    }
    usageOptions.append(explanation + "\n");
    if(mandatory) {
	shortRequiredArgs.append(std::string(1, optionName));
    }
}

/**
  * @brief Reads and tests program short or long arguments and returns a map with short name of program arguments as map keys
  * and parameter value of program arguments as map values.
  * @param[in] argc number of program arguments (incluiding program name)
  * @param[in] argv array of program arguments as strings
  * @param[in] longOptions struct with long names for options
  * @param[in] shortOptions string with a letter for every supported option
  * @param[in] shortRequiredArgs string with a letter for every required option
  * @param[in] usage string with text explaining correct program invocation
  * @return map with short program option name as a key and program option value as map value
  */
ProgramConfig::ExecArgsMap ProgramConfig::readProgramArguments(int argc, char* argv[], struct option longOptions[],
	std::string shortOptions, std::string shortRequiredArgs,
	std::string usage) {
    ExecArgsMap resultMap;

    resultMap["programName"] = argv[0];
    try {
         int option_index = 0;
	while(1) {
	   int option;
	    option = getopt_long(argc, argv, shortOptions.c_str(),
				 longOptions, &option_index);
	    switch(option) {
		case -1: // End of  params
		    return resultMap;
		case '?': /* getopt_long already printed an error message. */
		    exit(-1);
		    break;
		default:
		    PROGRAMCONFIG_CERR("option_index: " << option_index
				       << " long_option: " << longOptions[option_index].name
				       << " option: " << (char) option << " optopt: " << (char) optopt
				       << " optarg: " << optarg << std::endl);
		    resultMap[std::string(1, (char)option)] = optarg;
		    break;
	    }
	}
    }
    catch(const std::invalid_argument& ex) {
	BTTHROW(std::invalid_argument(std::string(__FILE__) +
				    ": " + std::to_string(__LINE__) + ": " +
				    ex.what() +
				    ": Error reading program parameters"), "ProgramConfig::readProgramArguments");
    }

    return resultMap;
}

/**
  * @brief Reads and tests program short arguments and returns a map with short name of program arguments as map keys and
  * parameter value of program arguments as map values.
  * @param[in] argc number of program arguments (incluiding program name)
  * @param[in] argv array of program arguments as strings
  * @param[in] shortArgs string with a letter for every supported option and tags for required and optional option arguments
  * (in standard getopt function format)
  * @param[in] shortRequiredArgs string with a letter for every required option of the supported options
  * @param[in] usage string with text explaining correct program invocation
  * @return map with program arguments short name as map keys and program arguments parameter value as map values
  */
ProgramConfig::ExecArgsMap ProgramConfig::readProgramShortArguments(int argc, char* argv[], std::string shortArgs, std::string shortRequiredArgs,
	std::string usage, std::vector<std::string>* pNonOptionArgs) {
    ExecArgsMap resultMap;
    resultMap["programName"] = fileName(argv[0]);
    //opterr = 0; // Avoids sending message with error to stderr if option is not supported
    try {
        int option;
	while((option = getopt(argc, argv, shortArgs.c_str())) != -1) {
	    switch(option) {
		case '?':
		case 'h':
		    /* getopt_long already printed an error message. */
		    std::cerr << "Usage: " << fileName(std::string(argv[0])) << " " << usage << std::endl;
		    exit(-1);
		    break;
		default:
		    if(optarg == nullptr)
			resultMap[std::string(1, (char)option)] = "";
		    else
			resultMap[std::string(1, (char)option)] = optarg;
		    // Remove option charater from string of required arguments characers, if present
		    if(shortRequiredArgs != "") {
			PROGRAMCONFIG_CERR("option read: '" << ((char) option) << "', requiredArgs before erase: " << shortRequiredArgs << std::endl);
			size_t requiredArgPos = shortRequiredArgs.find((char) option);
			if(requiredArgPos != std::string::npos) {
			    shortRequiredArgs.erase(requiredArgPos, 1);
			    PROGRAMCONFIG_CERR("requiredArgs after erase: " << shortRequiredArgs << std::endl);
			}
			else {
			    PROGRAMCONFIG_CERR("option '" << ((char) option) << "' not in requiredArgs" << std::endl);
			}
		    }
	    }
	}
	PROGRAMCONFIG_CERR("first non-option argument (optind): " << optind << std::endl);
	// Add non-option arguments to nonOptionsArgs parameter
	for (auto i = optind; i < argc; i++) {
		pNonOptionArgs->push_back(argv[i]);
		PROGRAMCONFIG_CERR("non-option argument (" << i << "): " << argv[i] << std::endl);
	}

	if(shortRequiredArgs.size() != 0) {
	    std::cerr << "Missing required arg(s): " << shortRequiredArgs << std::endl;
	    std::cerr << "Usage: " << fileName(std::string(argv[0])) << " " << usage << std::endl;
	    exit(-1);
	}
    }
    catch(const std::invalid_argument& ex) {
	BTTHROW(std::invalid_argument(std::string(__FILE__) +
				    ": " + std::to_string(__LINE__) + ": " +
				    ex.what() +
				    ": Error reading program parameters"), "ProgramConfig::readProgramShortArguments");
    }
    return resultMap;
}

/**
 * @brief Returns the file name part from a path
 *
 * @param[in] path absolute or relative path of a file name
 * @return string with the file name part of the path
 */
std::string ProgramConfig::fileName(std::string path) {
    std::size_t found = path.find_last_of("/");
    if(found == std::string::npos) {
	return path;
    }
    else {
	return path.substr(found + 1);
    }
}

/**
* @brief get absolute pathname of the caller process' executable
* @return absolute pathname
*/
std::string ProgramConfig::exeFile() {
    struct stat statBuffer;
    if(::lstat(PROC_SELF_EXE, &statBuffer) == -1)
	BTTHROW(std::system_error(std::make_error_code(static_cast<std::errc>(errno)), "Unable to stat " PROC_SELF_EXE),"ProgramConfig::exeFilePath");
    ssize_t nameLen = statBuffer.st_size;

    // lstat may return 0 length for special files
    if(nameLen == 0)
	nameLen = PATH_MAX;

    auto exePath = new char[nameLen +1];	// +1 to check for truncation at return value from readlink
    ssize_t bytesRead = ::readlink(PROC_SELF_EXE, exePath, nameLen);
    if(bytesRead == -1) {
	BTTHROW(std::system_error(std::make_error_code(static_cast<std::errc>(errno)), "Unable to read link " PROC_SELF_EXE),"ProgramConfig::exeFilePath");
    }
    else if(bytesRead == nameLen)
	BTTHROW(std::system_error(std::make_error_code(static_cast<std::errc>(ENAMETOOLONG)), "Unable to read link " PROC_SELF_EXE),"ProgramConfig::exeFilePath");

    return std::string(exePath, bytesRead);
}

/**
* @brief get absolute directory that contains the caller process' executable
* @return absolute directory name
*/
std::string ProgramConfig::exeDir() {
    std::string exePath = exeFile();

    std::size_t found = exePath.find_last_of("/");
    if(found == std::string::npos)
	return std::string("/");	// This should never happen (/proc/self/exe must have one slash at least)
    else
	return exePath.substr(0, found);
}

/**
* @brief method for getting usage text from usageSummary and usageOption class fields
* @return usage text
*/
std::string ProgramConfig::ConfigTraits::getUsage() {
    return usageSummary + "\n" + usageOptions;
}

/**
 * @brief Reads program arguments and set, depending on their values, the fields of the configuration object
 * pConfigTraits.
 *
 * It calls readProgramShortArguments method (that reads and tests program short arguments and returns a map with
 * short name of program arguments as map keys and parameter value of program arguments as map values) and calls
 * setConfig pure virtual method (that reads map with program arguments information and sets, depending on its values,
 * fields of the class variable which stores configuration information, pConfigTraits).
 *
 * @param[in] argc number of program arguments (incluiding program name)
 * @param[in] argv array of program arguments as strings
 */
void ProgramConfig::readExecArgs(int argc, char* argv[]) {
    pConfigTraits->execArgsMap =
	ProgramConfig::readProgramShortArguments(argc, argv, pConfigTraits->shortArgs,
		pConfigTraits->shortRequiredArgs,
		pConfigTraits->getUsage(), &(pConfigTraits->nonOptionArgs));
}

/**
* @brief check if all required run arguments are present (exits program if false)
*/
void ProgramConfig::checkRequiredArgsPresent() {
    if(pConfigTraits->shortRequiredArgs.size() != 0) {
	std::cerr << "Missing required arg(s): " << pConfigTraits->shortRequiredArgs << std::endl;
	std::cerr << "Usage: " << pConfigTraits->programName << " " << pConfigTraits->getUsage() << std::endl;
	exit(-1);
    }
}

/**
* @brief method for setting class fields according to run parameters read
*/
void ProgramConfig::ConfigTraits::configure() {
    // Visible ConfigTraits struct is this class ConfigTraits
    ExecArgsMap::iterator it;
    it = execArgsMap.find("programName");
    if(it != execArgsMap.end()) {
	programName = it->second;
	execArgsMap.erase("programName");
    }
}
}
#undef PROGRAMCONFIG_DEBUG

