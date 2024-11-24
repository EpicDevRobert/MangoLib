# Minimalna wersja CMake
cmake_minimum_required(VERSION 3.10)

# Nazwa projektu
project(MangoLib)

# Standard C++ (zaktualizuj do C++14 lub wy¿szego, jeœli to mo¿liwe)
set(CMAKE_CXX_STANDARD 14)

# Folder wyjœciowy dla bibliotek i binarek
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)   # Pliki statyczne (.a)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)   # Pliki dynamiczne (.dll)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)   # Pliki wykonywalne (.exe)

# Dodanie Ÿróde³ biblioteki (pliki .cpp)
add_library(MangoLib SHARED
    "Src/mango.h"
    Src/MangoStandard/standard.cpp
    "Src/DeepLearningMango/DeepMango.cpp" 
    Src/Aktywacja/Aktywacje.cpp
 "Src/DeepLearningMango/DeepMango.h")

# Dodanie nag³ówków do interfejsu publicznego
target_include_directories(MangoLib PUBLIC
    ${PROJECT_SOURCE_DIR}/Src
    ${PROJECT_SOURCE_DIR}/Src/MangoStandard
    ${PROJECT_SOURCE_DIR}/Src/Aktywacja
)

# Regu³y instalacji
install(TARGETS MangoLib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Instalacja nag³ówków
install(DIRECTORY ${PROJECT_SOURCE_DIR}/Src/
    DESTINATION include/MangoLib
    FILES_MATCHING PATTERN "*.h"
)
