import os

from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy, get, replace_in_file


class Open3DConan(ConanFile):
    """Open3D is not on Conan Center, so this local recipe builds it from
    source (exported to the cache automatically by scikit-build-core-conan's
    ``local-recipes`` hook — see pyproject.toml). Version is supplied at
    export time (``--version``) and must have an entry in conandata.yml.

    Minimal static build: no GUI (skips filament), no WebRTC, no Python
    module, no examples — just the core geometry/pipelines needed for 3D
    registration. Linux x86_64 note: Open3D's default BLAS is a downloaded
    static Intel MKL (USE_BLAS=OFF); building OpenBLAS instead
    (use_blas=True) avoids the MKL license but requires gfortran.
    """

    name = "open3d"
    package_type = "static-library"
    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "use_blas": [True, False],
        # X11 support in the bundled GLFW (only matters for the legacy
        # native visualizer). Off by default: it needs the X11 extension
        # dev headers (apt: libxrandr-dev libxinerama-dev libxcursor-dev
        # libxi-dev); without it GLFW builds its "null" platform — all of
        # Open3D works except opening visualizer windows from C++.
        "with_x11": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "use_blas": False,
        "with_x11": False,
    }

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def requirements(self):
        # Open3D fetches a pinned Eigen commit from gitlab.com, which 403s
        # from some networks; Conan Center's eigen avoids that (and Open3D
        # officially supports system Eigen ≥ 3.4). transitive_headers: Eigen
        # appears in Open3D's public headers, and the installed Open3DConfig
        # does find_dependency(Eigen3) at consumer time.
        self.requires("eigen/3.4.0", transitive_headers=True)

    def validate(self):
        check_min_cppstd(self, 17)

    def layout(self):
        cmake_layout(self)

    def source(self):
        get(self, **self.conan_data["sources"][self.version], strip_root=True)
        # Open3D passes no backend flags to the bundled GLFW 3.4, whose
        # defaults (Wayland + X11 on) require wayland-scanner and the X11
        # extension dev headers. Route the backend choice through CMake
        # variables set from recipe options in generate() (source() itself
        # must stay option-independent).
        replace_in_file(
            self,
            os.path.join(self.source_folder, "3rdparty", "glfw", "glfw.cmake"),
            "-DGLFW_BUILD_TESTS=OFF",
            "-DGLFW_BUILD_TESTS=OFF\n"
            "        -DGLFW_BUILD_WAYLAND=OFF\n"
            "        -DGLFW_BUILD_X11=${GLFW_BUILD_X11}",
        )

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        # Apart from Eigen (Conan, see requirements), Open3D's own build
        # fetches its bundled 3rdparty deps (fmt, embree, VTK, MKL/OpenBLAS,
        # ...) at build time.
        tc.cache_variables["USE_SYSTEM_EIGEN3"] = True
        # Consumed by the injected flags in 3rdparty/glfw/glfw.cmake.
        tc.cache_variables["GLFW_BUILD_X11"] = bool(self.options.with_x11)
        tc.cache_variables["BUILD_EXAMPLES"] = False
        tc.cache_variables["BUILD_PYTHON_MODULE"] = False
        tc.cache_variables["BUILD_UNIT_TESTS"] = False
        tc.cache_variables["BUILD_BENCHMARKS"] = False
        tc.cache_variables["BUILD_GUI"] = False
        tc.cache_variables["BUILD_WEBRTC"] = False
        # ISPC-vectorized kernels need the ISPC compiler download; off for now.
        tc.cache_variables["BUILD_ISPC_MODULE"] = False
        tc.cache_variables["USE_BLAS"] = bool(self.options.use_blas)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        copy(
            self,
            "LICENSE",
            src=self.source_folder,
            dst=os.path.join(self.package_folder, "licenses"),
        )
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        # Open3D installs its own CMake config (which knows how to link the
        # bundled 3rdparty static libs), so skip CMakeDeps generation and let
        # find_package(Open3D) discover it through CMAKE_PREFIX_PATH.
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.builddirs = [".", os.path.join("lib", "cmake", "Open3D")]
