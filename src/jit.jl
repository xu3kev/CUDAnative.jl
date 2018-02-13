# JIT compilation of Julia code to PTX

export cufunction


#
# main code generation functions
#

# make function names safe for PTX
safe_fn(fn::String) = replace(fn, r"[^aA-zZ0-9_]"=>"_")
safe_fn(f::Core.Function) = safe_fn(String(typeof(f).name.mt.name))
safe_fn(f::LLVM.Function) = safe_fn(LLVM.name(f))

function raise_exception(insblock::BasicBlock, ex::Value)
    fun = LLVM.parent(insblock)
    mod = LLVM.parent(fun)
    ctx = context(mod)

    builder = Builder(ctx)
    position!(builder, insblock)

    trap = if haskey(functions(mod), "llvm.trap")
        functions(mod)["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", LLVM.FunctionType(LLVM.VoidType(ctx)))
    end
    call!(builder, trap)
end

function irgen(@nospecialize(f), @nospecialize(tt))
    # get the method instance
    world = typemax(UInt)
    meth = which(f, tt)
    sig_tt = Tuple{typeof(f), tt.parameters...}
    (ti, env) = ccall(:jl_type_intersection_with_env, Any,
                      (Any, Any), sig_tt, meth.sig)::Core.SimpleVector
    meth = Base.func_for_method_checked(meth, ti)
    linfo = ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
                  (Any, Any, Any, UInt), meth, ti, env, world)

    # set-up the compiler interface
    function hook_raise_exception(insblock::Ptr{Cvoid}, ex::Ptr{Cvoid})
        insblock = convert(LLVM.API.LLVMValueRef, insblock)
        ex = convert(LLVM.API.LLVMValueRef, ex)
        raise_exception(BasicBlock(insblock), Value(ex))
    end
    params = Base.CodegenParams(track_allocations=false,
                                code_coverage=false,
                                static_alloc=false,
                                prefer_specsig=true,
                                raise_exception=hook_raise_exception)

    # generate IR
    native_code = ccall(:jl_create_native, Ptr{Cvoid},
                        (Vector{Core.MethodInstance}, Base.CodegenParams),
                        [linfo], params)
    @assert native_code != C_NULL
    llvm_mod_ref = ccall(:jl_get_llvm_module, LLVM.API.LLVMModuleRef,
                         (Ptr{Cvoid},), native_code)
    @assert llvm_mod_ref != C_NULL
    llvm_mod = LLVM.Module(llvm_mod_ref)

    # get the top-level function index
    api = Ref{UInt8}(typemax(UInt8))
    llvm_func_idx = Ref{UInt32}()
    llvm_specfunc_idx = Ref{UInt32}()
    ccall(:jl_get_function_id, Nothing,
          (Ptr{Cvoid}, Ptr{Core.MethodInstance}, Ptr{UInt8}, Ptr{UInt32}, Ptr{UInt32}),
          native_code, Ref(linfo), api, llvm_func_idx, llvm_specfunc_idx)
    @assert api[] != typemax(api[])

    # get the top-level function
    llvm_func_ref = ccall(:jl_get_llvm_function, LLVM.API.LLVMValueRef,
                     (Ptr{Cvoid}, UInt32), native_code, llvm_func_idx[]-1)
    @assert llvm_func_ref != C_NULL
    llvm_func = LLVM.Function(llvm_func_ref)
    llvm_specfunc_ref = ccall(:jl_get_llvm_function, LLVM.API.LLVMValueRef,
                         (Ptr{Cvoid}, UInt32), native_code, llvm_specfunc_idx[]-1)
    @assert llvm_specfunc_ref != C_NULL
    llvm_specfunc = LLVM.Function(llvm_specfunc_ref)

    # configure the module
    # NOTE: NVPTX::TargetMachine's data layout doesn't match the NVPTX user guide,
    #       so we specify it ourselves
    if Int === Int64
        triple!(llvm_mod, "nvptx64-nvidia-cuda")
        datalayout!(llvm_mod, "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64")
    else
        triple!(llvm_mod, "nvptx-nvidia-cuda")
        datalayout!(llvm_mod, "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64")
    end

    # clean up incompatibilities
    for llvm_func in functions(llvm_mod)
        # remove non-specsig functions
        fn = LLVM.name(llvm_func)
        if startswith(fn, "jlcall_")
            unsafe_delete!(llvm_mod, llvm_func)
            continue
        end

        # only occurs in debug builds
        delete!(function_attributes(llvm_func), EnumAttribute("sspreq", 0, jlctx[]))

        # make function names safe for ptxas
        # (LLVM ought to do this, see eg. D17738 and D19126), but fails
        # TODO: fix all globals?
        if !isdeclaration(llvm_func)
            fn = safe_fn(llvm_func)
            if fn != fn
                LLVM.name!(llvm_func, fn)
            end
        end
    end

    return llvm_mod, llvm_specfunc
end

# promote a function to a kernel
function promote_kernel!(mod::LLVM.Module, entry_f::LLVM.Function, @nospecialize(tt);
                         minthreads::Union{Nothing,CuDim}=nothing,
                         maxthreads::Union{Nothing,CuDim}=nothing,
                         blocks_per_sm::Union{Nothing,Integer}=nothing,
                         maxregs::Union{Nothing,Integer}=nothing)
    kernel = wrap_entry!(mod, entry_f, tt);


    # property annotations

    annotations = LLVM.Value[kernel]

    ## kernel metadata
    append!(annotations, [MDString("kernel"), ConstantInt(Int32(1))])

    ## expected CTA sizes
    for (typ,vals) in (:req=>minthreads, :max=>maxthreads)
        if vals != nothing
            bounds = CUDAdrv.CuDim3(vals)
            for dim in (:x, :y, :z)
                bound = getfield(bounds, dim)
                append!(annotations, [MDString("$(typ)ntid$(dim)"),
                                      ConstantInt(Int32(bound))])
            end
        end
    end

    if blocks_per_sm != nothing
        append!(annotations, [MDString("minctasm"), ConstantInt(Int32(blocks_per_sm))])
    end

    if maxregs != nothing
        append!(annotations, [MDString("maxnreg"), ConstantInt(Int32(maxregs))])
    end


    push!(metadata(mod), "nvvm.annotations", MDNode(annotations))


    return kernel
end

# maintain our own "global unique" suffix for disambiguating kernels
globalUnique = 0

# generate a kernel wrapper to fix & improve argument passing
function wrap_entry!(mod::LLVM.Module, entry_f::LLVM.Function, @nospecialize(tt))
    entry_ft = eltype(llvmtype(entry_f))
    @assert return_type(entry_ft) == LLVM.VoidType(jlctx[])

    # filter out ghost types, which don't occur in the LLVM function signatures
    julia_types = filter(dt->!isghosttype(dt), tt.parameters)

    # generate the wrapper function type & def
    global globalUnique
    function wrapper_type(julia_t, codegen_t)
        if isa(codegen_t, LLVM.PointerType) && !(julia_t <: Ptr)
            # we didn't specify a pointer, but codegen passes one anyway.
            # make the wrapper accept the underlying value instead.
            return eltype(codegen_t)
        else
            return codegen_t
        end
    end
    wrapper_types = LLVM.LLVMType[wrapper_type(julia_t, codegen_t)
                                  for (julia_t, codegen_t)
                                  in zip(julia_types, parameters(entry_ft))]
    wrapper_fn = "ptxcall" * LLVM.name(entry_f)[6:end]
    wrapper_fn = replace(wrapper_fn, r"\d+$" => (globalUnique+=1))
    wrapper_ft = LLVM.FunctionType(LLVM.VoidType(jlctx[]), wrapper_types)
    wrapper_f = LLVM.Function(mod, wrapper_fn, wrapper_ft)

    # emit IR performing the "conversions"
    Builder(jlctx[]) do builder
        entry = BasicBlock(wrapper_f, "entry", jlctx[])
        position!(builder, entry)

        wrapper_args = Vector{LLVM.Value}()

        # perform argument conversions
        codegen_types = parameters(entry_ft)
        wrapper_params = parameters(wrapper_f)
        for (julia_t, codegen_t, wrapper_t, wrapper_param) in
            zip(julia_types, codegen_types, wrapper_types, wrapper_params)
            if codegen_t != wrapper_t
                # the wrapper argument doesn't match the kernel parameter type.
                # this only happens when codegen wants to pass a pointer.
                @assert isa(codegen_t, LLVM.PointerType)
                @assert eltype(codegen_t) == wrapper_t

                # copy the argument value to a stack slot, and reference it.
                ptr = alloca!(builder, wrapper_t)
                if LLVM.addrspace(codegen_t) != 0
                    ptr = addrspacecast!(builder, ptr, codegen_t)
                end
                store!(builder, wrapper_param, ptr)
                push!(wrapper_args, ptr)

                # Julia marks parameters as TBAA immutable;
                # this is incompatible with us storing to a stack slot, so clear TBAA
                # TODO: tag with alternative information (eg. TBAA, or invariant groups)
                entry_params = collect(parameters(entry_f))
                candidate_uses = []
                for param in entry_params
                    append!(candidate_uses, collect(uses(param)))
                end
                while !isempty(candidate_uses)
                    usepair = popfirst!(candidate_uses)
                    inst = user(usepair)

                    md = metadata(inst)
                    if haskey(md, LLVM.MD_tbaa)
                        delete!(md, LLVM.MD_tbaa)
                    end

                    # follow along certain pointer operations
                    if isa(inst, LLVM.GetElementPtrInst) ||
                       isa(inst, LLVM.BitCastInst) ||
                       isa(inst, LLVM.AddrSpaceCastInst)
                        append!(candidate_uses, collect(uses(inst)))
                    end
                end
            else
                push!(wrapper_args, wrapper_param)
            end
        end

        call!(builder, entry_f, wrapper_args)

        ret!(builder)
    end

    # early-inline the original entry function into the wrapper
    push!(function_attributes(entry_f), EnumAttribute("alwaysinline", 0, jlctx[]))
    linkage!(entry_f, LLVM.API.LLVMInternalLinkage)
    ModulePassManager() do pm
        always_inliner!(pm)
        run!(pm, mod)
    end

    return wrapper_f
end

const libdevices = Dict{String, LLVM.Module}()
function link_libdevice!(mod::LLVM.Module, cap::VersionNumber)
    CUDAnative.configured || return

    # find libdevice
    path = if isa(libdevice, Dict)
        # select the most recent & compatible library
        vers = keys(CUDAnative.libdevice)
        compat_vers = Set(ver for ver in vers if ver <= cap)
        isempty(compat_vers) && error("No compatible CUDA device library available")
        ver = maximum(compat_vers)
        path = libdevice[ver]
    else
        libdevice
    end

    # load the library, once
    if !haskey(libdevices, path)
        open(path) do io
            libdevice_mod = parse(LLVM.Module, read(io), jlctx[])
            name!(libdevice_mod, "libdevice")
            libdevices[path] = libdevice_mod
        end
    end
    libdevice_mod = LLVM.Module(libdevices[path])

    # override libdevice's triple and datalayout to avoid warnings
    triple!(libdevice_mod, triple(mod))
    datalayout!(libdevice_mod, datalayout(mod))

    # 1. save list of external functions
    exports = map(LLVM.name, functions(mod))
    filter!(fn->!haskey(functions(libdevice_mod), fn), exports)

    # 2. link with libdevice
    link!(mod, libdevice_mod)

    ModulePassManager() do pm
        # 3. internalize all functions not in list from (1)
        internalize!(pm, exports)

        # 4. eliminate all unused internal functions
        #
        # this isn't necessary, as we do the same in optimize! to inline kernel wrappers,
        # but it results _much_ smaller modules which are easier to handle on optimize=false
        global_optimizer!(pm)
        global_dce!(pm)
        strip_dead_prototypes!(pm)

        # 5. run NVVMReflect pass
        push!(metadata(mod), "nvvm-reflect-ftz",
              MDNode([ConstantInt(Int32(1))]))

        # 6. run standard optimization pipeline
        #
        #    see `optimize!`

        run!(pm, mod)
    end
end

function machine(cap::VersionNumber, triple::String)
    InitializeNVPTXTarget()
    InitializeNVPTXTargetInfo()
    t = Target(triple)

    InitializeNVPTXTargetMC()
    cpu = "sm_$(cap.major)$(cap.minor)"
    if cuda_driver_version >= v"9.0" && v"6.0" in ptx_support
        # in the case of CUDA 9, we use sync intrinsics from PTX ISA 6.0+
        feat = "+ptx60"
    else
        feat = ""
    end
    tm = TargetMachine(t, triple, cpu, feat)

    return tm
end

# Optimize a bitcode module according to a certain device capability.
function optimize!(mod::LLVM.Module, entry::LLVM.Function, cap::VersionNumber)
    tm = machine(cap, triple(mod))

    # GPU code is _very_ sensitive to register pressure and local memory usage,
    # so forcibly inline every function definition into the entry point
    # and internalize all other functions (enabling ABI-breaking optimizations).
    # FIXME: this is too coarse. use a proper inliner tuned for GPUs
    ModulePassManager() do pm
        if VERSION >= v"0.7.0-DEV.3650"
            no_inline = EnumAttribute("noinline", 0, jlctx[])
            always_inline = EnumAttribute("alwaysinline", 0, jlctx[])
            for f in filter(f->f!=entry && !isdeclaration(f), functions(mod))
                attrs = function_attributes(f)
                if !(no_inline in collect(attrs))
                    push!(attrs, always_inline)
                end
                linkage!(f, LLVM.API.LLVMInternalLinkage)
            end
            always_inliner!(pm)
        else
            # bugs and missing features prevent this from working on older Julia versions
            internalize!(pm, [LLVM.name(entry)])
        end
        run!(pm, mod)
    end

    ModulePassManager() do pm
        if Base.VERSION >= v"0.7.0-DEV.1494"
            add_library_info!(pm, triple(mod))
            add_transform_info!(pm, tm)
            ccall(:jl_add_optimization_passes, Cvoid,
                  (LLVM.API.LLVMPassManagerRef, Cint),
                  LLVM.ref(pm), Base.JLOptions().opt_level)

            # CUDAnative's JIT internalizes non-inlined child functions, making it possible
            # to rewrite them (whereas the Julia JIT caches those functions exactly);
            # this opens up some more optimization opportunities
            dead_arg_elimination!(pm)   # parent doesn't use return value --> ret void
        else
            add_transform_info!(pm, tm)
            # TLI added by PMB
            ccall(:LLVMAddLowerGCFramePass, Cvoid,
                  (LLVM.API.LLVMPassManagerRef,), LLVM.ref(pm))
            ccall(:LLVMAddLowerPTLSPass, Cvoid,
                  (LLVM.API.LLVMPassManagerRef, Cint), LLVM.ref(pm), 0)

            always_inliner!(pm) # TODO: set it as the builder's inliner
            PassManagerBuilder() do pmb
                populate!(pm, pmb)
            end
        end

        global_optimizer!(pm)
        global_dce!(pm)
        strip_dead_prototypes!(pm)

        run!(pm, mod)
    end
end

function mcgen(mod::LLVM.Module, func::LLVM.Function, cap::VersionNumber)
    tm = machine(cap, triple(mod))

    InitializeNVPTXAsmPrinter()
    return String(emit(tm, mod, LLVM.API.LLVMAssemblyFile))
end

# Compile a function to PTX, returning the assembly and an entry point.
# Not to be used directly, see `cufunction` instead.
#
# The `kernel` argument indicates whether we are compiling a kernel entry-point function,
# in which case extra metadata needs to be attached.
function compile_function(@nospecialize(func), @nospecialize(tt), cap::VersionNumber;
                          kernel::Bool=true, kwargs...)
    ## high-level code generation (Julia AST)

    sig = "$(typeof(func).name.mt.name)($(join(tt.parameters, ", ")))"
    @debug("(Re)compiling $sig for capability $cap")

    check_invocation(func, tt; kernel=kernel)


    ## low-level code generation (LLVM IR)

    mod, entry = irgen(func, tt)
    if kernel
        entry = promote_kernel!(mod, entry, tt; kwargs...)
    end
    @trace("Module entry point: ", LLVM.name(entry))

    # link libdevice, if it might be necessary
    # TODO: should be more find-grained -- only matching functions actually in this libdevice
    if any(f->isdeclaration(f) && intrinsic_id(f)==0, functions(mod))
        link_libdevice!(mod, cap)
    end

    # optimize the IR (otherwise the IR isn't necessarily compatible)
    optimize!(mod, entry, cap)

    # validate generated IR
    errors = validate_ir(mod)
    if !isempty(errors)
        for e in errors
            warn("Encountered incompatible LLVM IR for $sig at capability $cap: ", e)
        end
        error("LLVM IR generated for $sig at capability $cap is not compatible")
    end


    ## machine code generation (PTX assembly)

    module_asm = mcgen(mod, entry, cap)

    return module_asm, LLVM.name(entry)
end

# check validity of a function invocation, specified by the generic function and a tupletype
function check_invocation(@nospecialize(func), @nospecialize(tt); kernel::Bool=false)
    sig = "$(typeof(func).name.mt.name)($(join(tt.parameters, ", ")))"

    # get the method
    ms = Base.methods(func, tt)
    isempty(ms)   && throw(ArgumentError("no method found for $sig"))
    length(ms)!=1 && throw(ArgumentError("no unique matching method for $sig"))
    m = first(ms)

    # emulate some of the specsig logic from codegen.cppto detect non-native CC functions
    # TODO: also do this for device functions (#87)
    isconcretetype(tt) || throw(ArgumentError("invalid call to device function $sig: passing abstract arguments"))
    m.isva && throw(ArgumentError("invalid device function $sig: is a varargs function"))

    # kernels can't return values
    if kernel
        rt = Base.return_types(func, tt)[1]
        if rt != Nothing
            throw(ArgumentError("$sig is not a valid kernel as it returns $rt"))
        end
    end
end

# (func::Function, tt::Type, cap::VersionNumber; kwargs...)
const compile_hook = Ref{Union{Nothing,Function}}(nothing)

# Main entry point for compiling a Julia function + argtypes to a callable CUDA function
function cufunction(dev::CuDevice, @nospecialize(func), @nospecialize(tt); kwargs...)
    CUDAnative.configured || error("CUDAnative.jl has not been configured; cannot JIT code.")
    @assert isa(func, Core.Function)

    # select a capability level
    dev_cap = capability(dev)
    compat_caps = filter(cap -> cap <= dev_cap, target_support)
    isempty(compat_caps) &&
        error("Device capability v$dev_cap not supported by available toolchain")
    cap = maximum(compat_caps)

    if compile_hook[] != nothing
        compile_hook[](func, tt, cap; kwargs...)
    end

    (module_asm, module_entry) = compile_function(func, tt, cap; kwargs...)

    # enable debug options based on Julia's debug setting
    jit_options = Dict{CUDAdrv.CUjit_option,Any}()
    if Base.JLOptions().debug_level == 1
        jit_options[CUDAdrv.GENERATE_LINE_INFO] = true
    elseif Base.JLOptions().debug_level >= 2
        jit_options[CUDAdrv.GENERATE_DEBUG_INFO] = true
    end
    cuda_mod = CuModule(module_asm, jit_options)
    cuda_fun = CuFunction(cuda_mod, module_entry)

    @debug begin
        attr = attributes(cuda_fun)
        bin_ver = VersionNumber(divrem(attr[CUDAdrv.FUNC_ATTRIBUTE_BINARY_VERSION],10)...)
        ptx_ver = VersionNumber(divrem(attr[CUDAdrv.FUNC_ATTRIBUTE_PTX_VERSION],10)...)
        regs = attr[CUDAdrv.FUNC_ATTRIBUTE_NUM_REGS]
        local_mem = attr[CUDAdrv.FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES]
        shared_mem = attr[CUDAdrv.FUNC_ATTRIBUTE_SHARED_SIZE_BYTES]
        constant_mem = attr[CUDAdrv.FUNC_ATTRIBUTE_CONST_SIZE_BYTES]
        """Compiled $func to PTX $ptx_ver for SM $bin_ver using $regs registers.
           Memory usage: $local_mem B local, $shared_mem B shared, $constant_mem B constant"""
    end

    return cuda_fun, cuda_mod
end

function init_jit()
    llvm_args = [
        # Program name; can be left blank.
        "",
        # Enable generation of FMA instructions to mimic behavior of nvcc.
        "--nvptx-fma-level=1",
    ]
    LLVM.API.LLVMParseCommandLineOptions(Int32(length(llvm_args)),
        [Base.unsafe_convert(Cstring, llvm_arg) for llvm_arg in llvm_args], C_NULL)
end
