#include <windows.h>
#include <filesystem>
#include <stdexcept>
#include <string>

// Runtime assets are explicitly external. No resource archive, extraction,
// hidden LocalAppData cache or fallback into the development environment.
void configure_external_runtime(const std::filesystem::path& directory) {
    const auto root = std::filesystem::absolute(directory);
    const auto bin = root / L"bin";
    if (!SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS))
        throw std::runtime_error("Cannot configure Windows DLL search directories");
    if (std::filesystem::is_directory(bin)) {
        if (!AddDllDirectory(bin.c_str()))
            throw std::runtime_error("Cannot register external runtime directory: " + bin.u8string());
        const auto ort = bin / L"onnxruntime.dll";
        if (std::filesystem::is_regular_file(ort) &&
            !LoadLibraryExW(ort.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS))
            throw std::runtime_error("Cannot load external ONNX Runtime: " + ort.u8string() +
                                     " (Windows error " + std::to_string(GetLastError()) + ")");
    }
    auto has_value = [](const wchar_t* name) { wchar_t value[2]{}; return GetEnvironmentVariableW(name, value, 2) != 0; };
    if (!has_value(L"NATIVE_NR_TOOLCHAIN") && !has_value(L"DLSS5_FP8_TOOLCHAIN")) {
        _wputenv_s(L"NATIVE_NR_TOOLCHAIN", (root / L"cuda12.8").c_str());
    }
}
