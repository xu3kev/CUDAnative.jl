var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#CUDAnative.jl-1",
    "page": "Home",
    "title": "CUDAnative.jl",
    "category": "section",
    "text": "Support for compiling and executing native Julia kernels on CUDA hardware.This package provides support for compiling and executing native Julia kernels on CUDA hardware. It is a work in progress, and only works on very recent versions of Julia ."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "Requirements:Julia 0.6 with LLVM 3.9 built from source, executed in tree (for LLVM.jl)\nNVIDIA driver, providing libcuda.so (for CUDAdrv.jl)\nCUDA toolkitAlthough that first requirement might sound complicated, it basically means you need to fetch and compile a copy of Julia 0.6 (refer to the main repository\'s README, checking out the latest tag for 0.6), and execute the resulting julia binary in-place without doing a make install. Afterwards, you can do:Pkg.add(\"CUDAnative\")\nusing CUDAnative\n\n# optionally\nPkg.test(\"CUDAnative\")For now, only Linux and macOS are supported. The build step will discover the available CUDA and LLVM installations, and figure out which devices can be programmed using that set-up. It depends on CUDAdrv and LLVM being properly configured.Even if the build fails, CUDAnative.jl should always be loadable. This simplifies use by downstream packages, until there is proper language support for conditional modules. You can check whether the package has been built properly by inspecting the CUDAnative.configured global variable."
},

{
    "location": "man/usage.html#",
    "page": "Usage",
    "title": "Usage",
    "category": "page",
    "text": ""
},

{
    "location": "man/usage.html#Usage-1",
    "page": "Usage",
    "title": "Usage",
    "category": "section",
    "text": ""
},

{
    "location": "man/usage.html#Quick-start-1",
    "page": "Usage",
    "title": "Quick start",
    "category": "section",
    "text": "First you have to write the kernel function and make sure it only uses features from the CUDA-supported subset of Julia:using CUDAnative\n\nfunction kernel_vadd(a, b, c)\n    i = (blockIdx().x-1) * blockDim().x + threadIdx().x\n    c[i] = a[i] + b[i]\n\n    return nothing\nend\nUsing the @cuda macro, you can launch the kernel on a GPU of your choice:using CUDAdrv, CUDAnative\nusing Base.Test\n\n# CUDAdrv functionality: generate and upload data\na = round.(rand(Float32, (3, 4)) * 100)\nb = round.(rand(Float32, (3, 4)) * 100)\nd_a = CuArray(a)\nd_b = CuArray(b)\nd_c = similar(d_a)  # output array\n\n# run the kernel and fetch results\n# syntax: @cuda (dims...) kernel(args...)\n@cuda (1,12) kernel_vadd(d_a, d_b, d_c)\n\n# CUDAdrv functionality: download data\n# this synchronizes the device\nc = Array(d_c)\n\n@test a+b ≈ cThis code is executed in a default, global context for the first device in your system. The compiler queries the context through CuCurrentContext, which implies you can easily switch contexts (using a different device, or supplying different flags) by activating a different one:dev = CuDevice(0)\nCuContext(dev) do ctx\n    # allocate things in this context\n    @cuda ...\nend"
},

{
    "location": "man/usage.html#Julia-support-1",
    "page": "Usage",
    "title": "Julia support",
    "category": "section",
    "text": "Only a limited subset of Julia is supported by this package. This subset is undocumented, as it is too much in flux.In general, GPU support of Julia code is determined by the language features used by the code. Several parts of the language are downright disallowed, such as calls to the Julia runtime, or garbage allocations. Other features might get reduced in strength, eg. throwing exceptions will result in a trap.If your code is incompatible with GPU execution, the compiler will mention the unsupported feature, and where the use came from:julia> foo(i) = (print(\"can\'t do this\"); return nothing)\nfoo (generic function with 1 method)\n\njulia> @cuda (1,1) foo(1)\nERROR: error compiling foo: error compiling print: generic call to unsafe_write requires the runtime language featureIn addition, the JIT doesn\'t support certain modes of compilation. For example, recursive functions require a proper cached compilation, which is currently absent."
},

{
    "location": "man/usage.html#CUDA-support-1",
    "page": "Usage",
    "title": "CUDA support",
    "category": "section",
    "text": "Not all of CUDA is supported, and because of time constraints the supported subset is again undocumented. The following (incomplete) list details the support and their CUDAnative.jl names. Most are implemented in intrinsics.jl, so have a look at that file for a more up to date list:Indexing: threadIdx().{x,y,z}, blockDim(), blockIdx(), gridDim(), warpsize()\nShared memory: @cuStaticSharedMemory, @cuDynamicSharedMemory\nArray type: CuDeviceArray (converted from input CuArrays, or shared memory)\nI/O: @cuprintf\nSynchronization: sync_threads\nCommunication: vote_{all,any,ballot}\nData movement: shfl_{up,down,bfly,idx}"
},

{
    "location": "man/usage.html#libdevice-1",
    "page": "Usage",
    "title": "libdevice",
    "category": "section",
    "text": "In addition to the native intrinsics listed above, math functionality from libdevice is wrapped and part of CUDAnative. For now, you need to fully qualify function calls to these intrinsics, which provide similar functionality to some of the low-level math functionality of Base which would otherwise call out to libm."
},

{
    "location": "man/troubleshooting.html#",
    "page": "Troubleshooting",
    "title": "Troubleshooting",
    "category": "page",
    "text": ""
},

{
    "location": "man/troubleshooting.html#Troubleshooting-1",
    "page": "Troubleshooting",
    "title": "Troubleshooting",
    "category": "section",
    "text": "You can enable verbose logging using two environment variables:DEBUG: if set, enable additional (possibly costly) run-time checks, and some more verbose output\nTRACE: if set, the DEBUG level will be activated, in addition with a trace of every call to the underlying libraryIn order to avoid run-time cost for checking the log level, these flags are implemented by means of global constants. As a result, you need to run Julia with precompilation disabled if you want to modify these flags:$ TRACE=1 julia --compilecache=no examples/vadd.jl\nTRACE: CUDAnative.jl is running in trace mode, this will generate a lot of additional output\n...Enabling colors with --color=yes is also recommended as it color-codes the output."
},

{
    "location": "man/troubleshooting.html#trap-and-kernel-launch-failures-1",
    "page": "Troubleshooting",
    "title": "trap and kernel launch failures",
    "category": "section",
    "text": "Exceptions, like the ones being thrown from out-of-bounds accesses, currently just generate a trap instruction which halts the GPU. This might show up as a kernel launch failure, or an unrelated error in another API call.If the error is thrown from an array access, and an out-of-bounds access is suspected, it is useful to turn of bounds checking (julia --check-bounds=no) and run the Julia process under cuda-memcheck while enabling debug mode 1 (the default value) or higher. This way, cuda-memcheck will be able to accurately pinpoint the out-of-bounds access, while specifying the exact location of the access within the active grid and block."
},

{
    "location": "man/troubleshooting.html#code_*-alternatives-1",
    "page": "Troubleshooting",
    "title": "code_* alternatives",
    "category": "section",
    "text": "CUDAnative provides alternatives to Base\'s code_llvm and code_native to inspect generated GPU code:julia> foo(a, i) = (a[1] = i; return nothing)\nfoo (generic function with 1 method)\n\njulia> a = CuArray{Int}(1)\n\njulia> CUDAnative.@code_llvm foo(a, 1)\n\n; Function Attrs: nounwind\ndefine i64 @julia_foo_62405(%CuDeviceArray.2* nocapture readonly, i64) {\n...\n}\n\njulia> @code_ptx foo(a, 1)\n.visible .entry julia_foo_62419(\n        .param .u64 julia_foo_62419_param_0,\n        .param .u64 julia_foo_62419_param_1\n)\n{\n...\n}\n\njulia> @code_sass foo(a, 1)\n        code for sm_20\n                Function : julia_foo_62539\n...Non-macro versions of these reflection entry-points are available as well (ie. code_llvm, etc), but as there\'s type conversions happening behind the scenes you will need to take care and perform those conversions manually:julia> CUDAnative.code_llvm(foo, (CuArray{Int,1},Int))\nERROR: error compiling foo: ...\n\njulia> CUDAnative.code_llvm(foo, (CuDeviceArray{Int,1},Int))\n\n; Function Attrs: nounwind\ndefine i64 @julia_foo_62405(%CuDeviceArray.2* nocapture readonly, i64) {\n...\n}"
},

{
    "location": "man/troubleshooting.html#Debug-info-and-line-number-information-1",
    "page": "Troubleshooting",
    "title": "Debug info and line-number information",
    "category": "section",
    "text": "LLVM\'s NVPTX back-end does not support the undocumented PTX debug format, so we cannot generate the necessary DWARF sections. This means that debugging generated code with e.g. cuda-gdb will be an unpleasant experience. Nonetheless, the PTX JIT is configured to emit debug info (which corresponds with nvcc -g) when the Julia debug info level is 2 or higher (julia -g2).We do however support emitting line number information, which is useful for other CUDA tools like cuda-memcheck. The functionality (which corresponds with nvcc -lineinfo) is enabled when the Julia debug info level is 1 (the default value) or higher."
},

{
    "location": "man/performance.html#",
    "page": "Performance",
    "title": "Performance",
    "category": "page",
    "text": ""
},

{
    "location": "man/performance.html#Performance-1",
    "page": "Performance",
    "title": "Performance",
    "category": "section",
    "text": "GPU code written in CUDAnative.jl can be as fast or even outperform CUDA C compiled with nvcc (on the condition that the same hardware features are used). This section will describe how to do so, and what to be careful about."
},

{
    "location": "man/performance.html#Profiling-1",
    "page": "Performance",
    "title": "Profiling",
    "category": "section",
    "text": "When optimizing code, it is important to know what to optimize. Luckily, the CUDA toolkit ships an excellent profiler, nvprof, with nvpp as the Eclipse-based UI. The CUDAnative compiler is fully compatible with these tools, and generates the required line number information to debug performance issues. To generate line number information, invoke Julia with the command-line option -g1. Using -g2 puts the PTX JIT in debug mode, which significantly lowers performance of GPU code and currently does not improve debugging.Although CUDAnative exports a @profile macro, it does not serve the same purpose as Base.@profile. Rather, it instructs the CUDA profiler to start right before the first kernel launch. This avoids profiling during the time Julia or CUDAnative precompile code, and result in a much more compact timeline view. If you want to use this feature, disable the nvprof/nvvp option to \"Start profiling at application start\". As with all Julia code, also perform a warm-up iteration without the profiler activated.For true source-level profiling akin to Base.@profile, look at nvvp\'s PC Sampling View (requires compute capability >= 5.2, CUDA >= 7.5). In the future, we might have a CUDAnative.@profile offering similar functionality, using the NVIDIA CUPTI library."
},

{
    "location": "man/performance.html#Optimizing-1",
    "page": "Performance",
    "title": "Optimizing",
    "category": "section",
    "text": "This section is a WIP. Some things to consider:Float64 is expensive, but literal floats are Float64. Use ...f0 or cast.\nSame for integers; although the performance hit is small, it increases register pressure."
},

{
    "location": "man/hacking.html#",
    "page": "Hacking",
    "title": "Hacking",
    "category": "page",
    "text": ""
},

{
    "location": "man/hacking.html#Hacking-1",
    "page": "Hacking",
    "title": "Hacking",
    "category": "section",
    "text": ""
},

{
    "location": "man/hacking.html#Generated-functions-1",
    "page": "Hacking",
    "title": "Generated functions",
    "category": "section",
    "text": "Generated functions are used heavily in CUDAnative.jl, in combination with LLVM.jl, to generate type-specialized code and IR. If evaluating the generator results in an error, Julia generates a dynamic call to the generator for you to inspect the error at run-time. This is a problem in the world of GPUs, where dynamic calls are prohibited. A band-aid is to print the exception during inference:diff --git a/base/inference.jl b/base/inference.jl\nindex 6443665676..b03d78ddaa 100644\n--- a/base/inference.jl\n+++ b/base/inference.jl\n@@ -2430,7 +2430,10 @@ function typeinf_frame(linfo::MethodInstance, caller, optimize::Bool, cached::Bo\n             try\n                 # user code might throw errors – ignore them\n                 src = get_staged(linfo)\n-            catch\n+            catch ex\n+                println(\"WARNING: An error occurred during generated function execution.\")\n+                println(ex)\n+                ccall(:jlbacktrace, Void, ())\n                 return nothing\n             end\n         else"
},

{
    "location": "man/hacking.html#Adding-intrinsics-1",
    "page": "Hacking",
    "title": "Adding intrinsics",
    "category": "section",
    "text": "Adding intrinsics to CUDAnative.jl can be relatively convoluted, depending on the type of intrinsic. Most of the boil down to inlining a snippet of LLVM IR, using llvmcall (or ccall with the llvmcall calling convention). For more complex code, use LLVM.jl to build the IR string."
},

{
    "location": "man/hacking.html#libdevice-intrinsics-1",
    "page": "Hacking",
    "title": "libdevice intrinsics",
    "category": "section",
    "text": "These intrinsics are represented by function calls to libdevice. Most of them should already be covered. There\'s a convenience macro, @wrap, simplifying the job of adding and exporting intrinsics, and converting arguments and return values. See the documentation of the macro for more details, and look at src/device/libdevice.jl for examples."
},

{
    "location": "man/hacking.html#LLVM-back-end-intrinsics-1",
    "page": "Hacking",
    "title": "LLVM back-end intrinsics",
    "category": "section",
    "text": "Calls to functions like llvm.nvvm.barrier0 are backed the PTX LLVM back-end, and can be wrapped using ccall with the llvmcall calling convention. For more complex intrinsics, or when you\'re not actually calling an intrinsic function, you can still use @wrap."
},

{
    "location": "man/hacking.html#Inline-PTX-assembly-1",
    "page": "Hacking",
    "title": "Inline PTX assembly",
    "category": "section",
    "text": "When there\'s no corresponding libdevice function or PTX back-end intrinsic exposing the required functionality, you can use inline PTX assembly via llvmcall. This requires you to embed the PTX assembly in LLVM IR, which is often messy.If the source of the assembly instructions is CUDA C code, you simplify this task by first compiling the CUDA code using Clang, and adapting the resulting LLVM IR for use within llvmcall. For example, extracting the following function definition from the CUDA SDK:__device__ unsigned int __ballot(int a)\n{\n  int result;\n  asm __volatile__ (\"{ \\n\\t\"\n        \".reg .pred \\t%%p1; \\n\\t\"\n        \"setp.ne.u32 \\t%%p1, %1, 0; \\n\\t\"\n        \"vote.ballot.b32 \\t%0, %%p1; \\n\\t\"\n        \"}\" : \"=r\"(result) : \"r\"(a));\n  return result;\n}We can generate the following LLVM IR by executing clang++ -Xclang -fcuda-is-device -S -emit-llvm -target nvptx64 ballot.cu -o - (you might need to add some CUDA boilerplate):define i32 @_Z8__balloti(i32 %a) #0 {\n  %1 = alloca i32, align 4\n  %result = alloca i32, align 4\n  store i32 %a, i32* %1, align 4\n  %2 = load i32, i32* %1, align 4\n  %3 = call i32 asm sideeffect \"{ \\0A\\09.reg .pred \\09%p1; \\0A\\09setp.ne.u32 \\09%p1, $1, 0; \\0A\\09vote.ballot.b32 \\09$0, %p1; \\0A\\09}\", \"=r,r\"(i32 %2) #1, !srcloc !1\n  store i32 %3, i32* %result, align 4\n  %4 = load i32, i32* %result, align 4\n  ret i32 %4\n}Finally, cleaning this code up we end up with the following llvmcall invocation:ballot_asm = \"\"\"{\n   .reg .pred %p1;\n   setp.ne.u32 %p1, \\$1, 0;\n   vote.ballot.b32 \\$0, %p1;\n}\"\"\"\n\nfunction ballot(pred::Bool)\n    return Base.llvmcall(\n        \"\"\"%2 = call i32 asm sideeffect \"$ballot_asm\", \"=r,r\"(i32 %0)\n           ret i32 %2\"\"\",\n        UInt32, Tuple{Int32}, convert(Int32, pred))\nendIn the future, we will use LLVM.jl to properly embed inline assembly instead of this string-based hackery."
},

{
    "location": "man/hacking.html#Other-functionality-1",
    "page": "Hacking",
    "title": "Other functionality",
    "category": "section",
    "text": "For other functionality, like shared memory, or when some additional management is required, like storing a global variable for printf\'s formatting string, you should use LLVM.jl to build the IR code instead of hacking strings together. As this doesn\'t touch global state, you can even do so from a @generated function. Do take care however to use Julia\'s LLVM context for all operations."
},

{
    "location": "lib/compilation.html#",
    "page": "Compilation & Execution",
    "title": "Compilation & Execution",
    "category": "page",
    "text": ""
},

{
    "location": "lib/compilation.html#CUDAnative.@cuda",
    "page": "Compilation & Execution",
    "title": "CUDAnative.@cuda",
    "category": "macro",
    "text": "@cuda (gridDim::CuDim, blockDim::CuDim, [shmem::Int], [stream::CuStream]) func(args...)\n\nHigh-level interface for calling functions on a GPU, queues a kernel launch on the current context. The gridDim and blockDim arguments represent the launch configuration, the optional shmem parameter specifies how much bytes of dynamic shared memory should be allocated (defaulting to 0), while the optional stream parameter indicates on which stream the launch should be scheduled.\n\nThe func argument should be a valid Julia function. It will be compiled to a CUDA function upon first use, and to a certain extent arguments will be converted and managed automatically (see cudaconvert). Finally, a call to cudacall is performed, scheduling the compiled function for execution on the GPU.\n\n\n\n"
},

{
    "location": "lib/compilation.html#CUDAnative.cudaconvert",
    "page": "Compilation & Execution",
    "title": "CUDAnative.cudaconvert",
    "category": "function",
    "text": "cudaconvert(x)\n\nThis function is called for every argument to be passed to a kernel, allowing it to be converted to a GPU-friendly format. By default, the function does nothing and returns the input object x as-is.\n\nFor CuArray objects, a corresponding CuDeviceArray object in global space is returned, which implements GPU-compatible array functionality.\n\n\n\n"
},

{
    "location": "lib/compilation.html#CUDAnative.nearest_warpsize",
    "page": "Compilation & Execution",
    "title": "CUDAnative.nearest_warpsize",
    "category": "function",
    "text": "Return the nearest number of threads that is a multiple of the warp size of a device:\n\nnearest_warpsize(dev::CuDevice, threads::Integer)\n\nThis is a common requirement, eg. when using shuffle intrinsics.\n\n\n\n"
},

{
    "location": "lib/compilation.html#Compilation-and-Execution-1",
    "page": "Compilation & Execution",
    "title": "Compilation & Execution",
    "category": "section",
    "text": "CUDAnative.@cuda\nCUDAnative.cudaconvert\nCUDAnative.nearest_warpsize"
},

{
    "location": "lib/reflection.html#",
    "page": "Reflection",
    "title": "Reflection",
    "category": "page",
    "text": ""
},

{
    "location": "lib/reflection.html#CUDAnative.code_llvm",
    "page": "Reflection",
    "title": "CUDAnative.code_llvm",
    "category": "function",
    "text": "code_llvm([io], f, types; optimize=true, dump_module=false, cap::VersionNumber)\n\nPrints the device LLVM IR generated for the method matching the given generic function and type signature to io which defaults to STDOUT. The IR is optimized according to optimize (defaults to true), and the entire module, including headers and other functions, is dumped if dump_module is set (defaults to false). The device capability cap to generate code for defaults to the current active device\'s capability, or v\"2.0\" if there is no such active context.\n\nSee also: @device_code_llvm, Base.code_llvm\n\n\n\n"
},

{
    "location": "lib/reflection.html#CUDAnative.code_ptx",
    "page": "Reflection",
    "title": "CUDAnative.code_ptx",
    "category": "function",
    "text": "code_ptx([io], f, types; cap::VersionNumber, kernel::Bool=false)\n\nPrints the PTX assembly generated for the method matching the given generic function and type signature to io which defaults to STDOUT. The device capability cap to generate code for defaults to the current active device\'s capability, or v\"2.0\" if there is no such active context. The optional kernel parameter indicates whether the function in question is an entry-point function, or a regular device function.\n\nSee also: @device_code_ptx\n\n\n\n"
},

{
    "location": "lib/reflection.html#CUDAnative.code_sass",
    "page": "Reflection",
    "title": "CUDAnative.code_sass",
    "category": "function",
    "text": "code_sass([io], f, types, cap::VersionNumber)\n\nPrints the SASS code generated for the method matching the given generic function and type signature to io which defaults to STDOUT. The device capability cap to generate code for defaults to the current active device\'s capability, or v\"2.0\" if there is no such active context. The method needs to be a valid entry-point kernel, eg. it should not return any values.\n\nSee also: @device_code_sass\n\n\n\n"
},

{
    "location": "lib/reflection.html#Reflection-1",
    "page": "Reflection",
    "title": "Reflection",
    "category": "section",
    "text": "Because of using a different compilation toolchain, CUDAnative.jl offers counterpart functions to the code_ functionality from Base:CUDAnative.code_llvm\nCUDAnative.code_ptx\nCUDAnative.code_sass"
},

{
    "location": "lib/reflection.html#CUDAnative.@device_code_lowered",
    "page": "Reflection",
    "title": "CUDAnative.@device_code_lowered",
    "category": "macro",
    "text": "@device_code_lowered ex\n\nEvaluates the expression ex and returns the result of Base.code_lowered for every compiled CUDA kernel.\n\nSee also: Base.@code_lowered\n\n\n\n"
},

{
    "location": "lib/reflection.html#CUDAnative.@device_code_typed",
    "page": "Reflection",
    "title": "CUDAnative.@device_code_typed",
    "category": "macro",
    "text": "@device_code_typed ex\n\nEvaluates the expression ex and returns the result of Base.code_typed for every compiled CUDA kernel.\n\nSee also: Base.@code_typed\n\n\n\n"
},

{
    "location": "lib/reflection.html#CUDAnative.@device_code_warntype",
    "page": "Reflection",
    "title": "CUDAnative.@device_code_warntype",
    "category": "macro",
    "text": "@device_code_warntype [io::IO=STDOUT] ex\n\nEvaluates the expression ex and prints the result of Base.code_warntype to io for every compiled CUDA kernel.\n\nSee also: Base.@code_warntype\n\n\n\n"
},

{
    "location": "lib/reflection.html#CUDAnative.@device_code_llvm",
    "page": "Reflection",
    "title": "CUDAnative.@device_code_llvm",
    "category": "macro",
    "text": "@device_code_llvm [io::IO=STDOUT] [optimize::Bool=true] [dump_module::Bool=false] ex\n\nEvaluates the expression ex and prints the result of Base.code_llvm to io for every compiled CUDA kernel. The optimize keyword argument determines whether the code is optimized, and dump_module can be used to print the entire LLVM module instead of only the entry-point function.\n\nSee also: Base.@code_llvm\n\n\n\n"
},

{
    "location": "lib/reflection.html#CUDAnative.@device_code_ptx",
    "page": "Reflection",
    "title": "CUDAnative.@device_code_ptx",
    "category": "macro",
    "text": "@device_code_ptx [io::IO=STDOUT] ex\n\nEvaluates the expression ex and prints the result of CUDAnative.code_ptx to io for every compiled CUDA kernel.\n\n\n\n"
},

{
    "location": "lib/reflection.html#CUDAnative.@device_code_sass",
    "page": "Reflection",
    "title": "CUDAnative.@device_code_sass",
    "category": "macro",
    "text": "@device_code_sass [io::IO=STDOUT] ex\n\nEvaluates the expression ex and prints the result of CUDAnative.code_sass to io for every compiled CUDA kernel.\n\n\n\n"
},

{
    "location": "lib/reflection.html#Convenience-macros-1",
    "page": "Reflection",
    "title": "Convenience macros",
    "category": "section",
    "text": "For ease of use, CUDAnative.jl also implements @device_code_ macros wrapping the above reflection functionality. These macros evaluate the expression argument, while tracing compilation and finally printing or returning the code for every invoked CUDA kernel. Do note that this evaluation can have side effects, as opposed to similarly-named @code_ macros in Base which are free of side effects.CUDAnative.@device_code_lowered\nCUDAnative.@device_code_typed\nCUDAnative.@device_code_warntype\nCUDAnative.@device_code_llvm\nCUDAnative.@device_code_ptx\nCUDAnative.@device_code_sass"
},

{
    "location": "lib/profiling.html#",
    "page": "Profiling",
    "title": "Profiling",
    "category": "page",
    "text": ""
},

{
    "location": "lib/profiling.html#CUDAnative.@profile",
    "page": "Profiling",
    "title": "CUDAnative.@profile",
    "category": "macro",
    "text": "@profile ex\n\nRuns your expression ex while activating the CUDA profiler upon first kernel launch. This makes it easier to profile accurately, without the overhead of initial compilation, memory transfers, ...\n\nNote that this API is used to programmatically control the profiling granularity by allowing profiling to be done only on selective pieces of code. It does not perform any profiling on itself, you need external tools for that.\n\n\n\n"
},

{
    "location": "lib/profiling.html#Profiling-1",
    "page": "Profiling",
    "title": "Profiling",
    "category": "section",
    "text": "CUDAnative.@profile"
},

{
    "location": "lib/device/intrinsics.html#",
    "page": "Intrinsics",
    "title": "Intrinsics",
    "category": "page",
    "text": ""
},

{
    "location": "lib/device/intrinsics.html#Intrinsics-1",
    "page": "Intrinsics",
    "title": "Intrinsics",
    "category": "section",
    "text": "This section lists the package\'s public functionality that corresponds to special CUDA functions to be used in device code. It is loosely organized according to the C language extensions appendix from the CUDA C programming guide. For more information about certain intrinsics, refer to the aforementioned NVIDIA documentation."
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.gridDim",
    "page": "Intrinsics",
    "title": "CUDAnative.gridDim",
    "category": "function",
    "text": "gridDim()::CuDim3\n\nReturns the dimensions of the grid.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.blockIdx",
    "page": "Intrinsics",
    "title": "CUDAnative.blockIdx",
    "category": "function",
    "text": "blockIdx()::CuDim3\n\nReturns the block index within the grid.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.blockDim",
    "page": "Intrinsics",
    "title": "CUDAnative.blockDim",
    "category": "function",
    "text": "blockDim()::CuDim3\n\nReturns the dimensions of the block.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.threadIdx",
    "page": "Intrinsics",
    "title": "CUDAnative.threadIdx",
    "category": "function",
    "text": "threadIdx()::CuDim3\n\nReturns the thread index within the block. \n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.warpsize",
    "page": "Intrinsics",
    "title": "CUDAnative.warpsize",
    "category": "function",
    "text": "warpsize()::UInt32\n\nReturns the warp size (in threads).\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#Indexing-and-Dimensions-1",
    "page": "Intrinsics",
    "title": "Indexing and Dimensions",
    "category": "section",
    "text": "CUDAnative.gridDim\nCUDAnative.blockIdx\nCUDAnative.blockDim\nCUDAnative.threadIdx\nCUDAnative.warpsize"
},

{
    "location": "lib/device/intrinsics.html#Memory-Types-1",
    "page": "Intrinsics",
    "title": "Memory Types",
    "category": "section",
    "text": ""
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.@cuStaticSharedMem",
    "page": "Intrinsics",
    "title": "CUDAnative.@cuStaticSharedMem",
    "category": "macro",
    "text": "@cuStaticSharedMem(typ::Type, dims) -> CuDeviceArray{typ,Shared}\n\nGet an array of type typ and dimensions dims (either an integer length or tuple shape) pointing to a statically-allocated piece of shared memory. The type should be statically inferable and the dimensions should be constant (without requiring constant propagation, see JuliaLang/julia#5560), or an error will be thrown and the generator function will be called dynamically.\n\nMultiple statically-allocated shared memory arrays can be requested by calling this macro multiple times.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.@cuDynamicSharedMem",
    "page": "Intrinsics",
    "title": "CUDAnative.@cuDynamicSharedMem",
    "category": "macro",
    "text": "@cuDynamicSharedMem(typ::Type, dims, offset::Integer=0) -> CuDeviceArray{typ,Shared}\n\nGet an array of type typ and dimensions dims (either an integer length or tuple shape) pointing to a dynamically-allocated piece of shared memory. The type should be statically inferable and the dimension and offset parameters should be constant (without requiring constant propagation, see JuliaLang/julia#5560), or an error will be thrown and the generator function will be called dynamically.\n\nDynamic shared memory also needs to be allocated beforehand, when calling the kernel.\n\nOptionally, an offset parameter indicating how many bytes to add to the base shared memory pointer can be specified. This is useful when dealing with a heterogeneous buffer of dynamic shared memory; in the case of a homogeneous multi-part buffer it is preferred to use view.\n\nNote that calling this macro multiple times does not result in different shared arrays; only a single dynamically-allocated shared memory array exists.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#Shared-Memory-1",
    "page": "Intrinsics",
    "title": "Shared Memory",
    "category": "section",
    "text": "CUDAnative.@cuStaticSharedMem\nCUDAnative.@cuDynamicSharedMem"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.sync_threads",
    "page": "Intrinsics",
    "title": "CUDAnative.sync_threads",
    "category": "function",
    "text": "sync_threads()\n\nWaits until all threads in the thread block have reached this point and all global and shared memory accesses made by these threads prior to sync_threads() are visible to all threads in the block.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.sync_warp",
    "page": "Intrinsics",
    "title": "CUDAnative.sync_warp",
    "category": "function",
    "text": "sync_warp(mask::Integer=0xffffffff)\n\nWaits threads in the warp, selected by means of the bitmask mask, have reached this point and all global and shared memory accesses made by these threads prior to sync_warp() are visible to those threads in the warp. The default value for mask selects all threads in the warp.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#Synchronization-1",
    "page": "Intrinsics",
    "title": "Synchronization",
    "category": "section",
    "text": "CUDAnative.sync_threads\nCUDAnative.sync_warp"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.vote_all",
    "page": "Intrinsics",
    "title": "CUDAnative.vote_all",
    "category": "function",
    "text": "vote_all(predicate::Bool)\n\nEvaluate predicate for all active threads of the warp and return non-zero if and only if predicate evaluates to non-zero for all of them.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.vote_any",
    "page": "Intrinsics",
    "title": "CUDAnative.vote_any",
    "category": "function",
    "text": "vote_any(predicate::Bool)\n\nEvaluate predicate for all active threads of the warp and return non-zero if and only if predicate evaluates to non-zero for any of them.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.vote_ballot",
    "page": "Intrinsics",
    "title": "CUDAnative.vote_ballot",
    "category": "function",
    "text": "vote_ballot(predicate::Bool)\n\nEvaluate predicate for all active threads of the warp and return an integer whose Nth bit is set if and only if predicate evaluates to non-zero for the Nth thread of the warp and the Nth thread is active.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#Warp-Vote-1",
    "page": "Intrinsics",
    "title": "Warp Vote",
    "category": "section",
    "text": "The warp vote functions allow the threads of a given warp to perform a reduction-and-broadcast operation. These functions take as input a boolean predicate from each thread in the warp and evaluate it. The results of that evaluation are combined (reduced) across the active threads of the warp in one different ways, broadcasting a single return value to each participating thread.CUDAnative.vote_all\nCUDAnative.vote_any\nCUDAnative.vote_ballot"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.shfl",
    "page": "Intrinsics",
    "title": "CUDAnative.shfl",
    "category": "function",
    "text": "shfl(val, lane::Integer, width::Integer=32)\n\nShuffle a value from a directly indexed lane lane.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.shfl_up",
    "page": "Intrinsics",
    "title": "CUDAnative.shfl_up",
    "category": "function",
    "text": "shfl_up(val, delta::Integer, width::Integer=32)\n\nShuffle a value from a lane with lower ID relative to caller.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.shfl_down",
    "page": "Intrinsics",
    "title": "CUDAnative.shfl_down",
    "category": "function",
    "text": "shfl_down(val, delta::Integer, width::Integer=32)\n\nShuffle a value from a lane with higher ID relative to caller.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.shfl_xor",
    "page": "Intrinsics",
    "title": "CUDAnative.shfl_xor",
    "category": "function",
    "text": "shfl_xor(val, mask::Integer, width::Integer=32)\n\nShuffle a value from a lane based on bitwise XOR of own lane ID with mask.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.shfl_sync",
    "page": "Intrinsics",
    "title": "CUDAnative.shfl_sync",
    "category": "function",
    "text": "shfl_sync(val, lane::Integer, width::Integer=32, threadmask::UInt32=0xffffffff)\n\nShuffle a value from a directly indexed lane lane. The default value for threadmask performs the shuffle on all threads in the warp.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.shfl_up_sync",
    "page": "Intrinsics",
    "title": "CUDAnative.shfl_up_sync",
    "category": "function",
    "text": "shfl_up_sync(val, delta::Integer, width::Integer=32, threadmask::UInt32=0xffffffff)\n\nShuffle a value from a lane with lower ID relative to caller. The default value for threadmask performs the shuffle on all threads in the warp.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.shfl_down_sync",
    "page": "Intrinsics",
    "title": "CUDAnative.shfl_down_sync",
    "category": "function",
    "text": "shfl_down_sync(val, delta::Integer, width::Integer=32, threadmask::UInt32=0xffffffff)\n\nShuffle a value from a lane with higher ID relative to caller. The default value for threadmask performs the shuffle on all threads in the warp.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.shfl_xor_sync",
    "page": "Intrinsics",
    "title": "CUDAnative.shfl_xor_sync",
    "category": "function",
    "text": "shfl_xor_sync(val, mask::Integer, width::Integer=32, threadmask::UInt32=0xffffffff)\n\nShuffle a value from a lane based on bitwise XOR of own lane ID with mask. The default value for threadmask performs the shuffle on all threads in the warp.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#Warp-Shuffle-1",
    "page": "Intrinsics",
    "title": "Warp Shuffle",
    "category": "section",
    "text": "CUDAnative.shfl\nCUDAnative.shfl_up\nCUDAnative.shfl_down\nCUDAnative.shfl_xorIf using CUDA 9.0, and PTX ISA 6.0 is supported, synchronizing versions of these intrinsics are available as well:CUDAnative.shfl_sync\nCUDAnative.shfl_up_sync\nCUDAnative.shfl_down_sync\nCUDAnative.shfl_xor_sync"
},

{
    "location": "lib/device/intrinsics.html#CUDAnative.@cuprintf",
    "page": "Intrinsics",
    "title": "CUDAnative.@cuprintf",
    "category": "macro",
    "text": "Print a formatted string in device context on the host standard output:\n\n@cuprintf(\"%Fmt\", args...)\n\nNote that this is not a fully C-compliant printf implementation; see the CUDA documentation for supported options and inputs.\n\nAlso beware that it is an untyped, and unforgiving printf implementation. Type widths need to match, eg. printing a 64-bit Julia integer requires the %ld formatting string.\n\n\n\n"
},

{
    "location": "lib/device/intrinsics.html#Formatted-Output-1",
    "page": "Intrinsics",
    "title": "Formatted Output",
    "category": "section",
    "text": "CUDAnative.@cuprintf"
},

{
    "location": "lib/device/array.html#",
    "page": "Arrays",
    "title": "Arrays",
    "category": "page",
    "text": ""
},

{
    "location": "lib/device/array.html#CUDAnative.CuDeviceArray",
    "page": "Arrays",
    "title": "CUDAnative.CuDeviceArray",
    "category": "type",
    "text": "CuDeviceArray(dims, ptr)\nCuDeviceArray{T}(dims, ptr)\nCuDeviceArray{T,A}(dims, ptr)\nCuDeviceArray{T,A,N}(dims, ptr)\n\nConstruct an N-dimensional dense CUDA device array with element type T wrapping a pointer, where N is determined from the length of dims and T is determined from the type of ptr. dims may be a single scalar, or a tuple of integers corresponding to the lengths in each dimension). If the rank N is supplied explicitly as in Array{T,N}(dims), then it must match the length of dims. The same applies to the element type T, which should match the type of the pointer ptr.\n\n\n\n"
},

{
    "location": "lib/device/array.html#Arrays-1",
    "page": "Arrays",
    "title": "Arrays",
    "category": "section",
    "text": "CUDAnative provides a primitive, lightweight array type to manage GPU data organized in an plain, dense fashion. This is the device-counterpart to CUDAdrv\'s CuArray, and implements (part of) the array interface as well as other functionality for use _on_ the GPU:CUDAnative.CuDeviceArray"
},

{
    "location": "lib/device/libdevice.html#",
    "page": "libdevice",
    "title": "libdevice",
    "category": "page",
    "text": ""
},

{
    "location": "lib/device/libdevice.html#libdevice-1",
    "page": "libdevice",
    "title": "libdevice",
    "category": "section",
    "text": "CUDAnative.jl provides wrapper functions for the mathematical routines in libdevice, CUDA\'s device math library. Many of these functions implement an interface familiar to similar functions in Base, but it is currently impossible to transparently dispatch to these device functions. As a consequence, users should prefix calls to math functions (eg. sin or pow) with the CUDAnative module name.WIP"
},

]}
