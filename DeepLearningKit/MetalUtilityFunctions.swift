//
//  MetalUtilFunctions.swift
//  MemkiteMetal
//
//  Created by Amund Tveit & Torb Morland on 24/11/15.
//  Copyright © 2015 Memkite. All rights reserved.
//

import Foundation
import Metal

func createComplexNumbersArray(_ count: Int) -> [MetalComplexNumberType] {
    let zeroComplexNumber = MetalComplexNumberType()
    return [MetalComplexNumberType](repeating: zeroComplexNumber, count: count)
}

public func createFloatNumbersArray(_ count: Int) -> [Float] {
    return [Float](repeating: 0.0, count: count)
}

func createFloatMetalBuffer(_ vector: [Float], metalDevice:MTLDevice) -> MTLBuffer {
    var vector = vector
    let byteLength = vector.count*sizeof(Float.self) // future: MTLResourceStorageModePrivate
    return metalDevice.newBuffer(withBytes: &vector, length: byteLength, options: MTLResourceOptions())
}

func createComplexMetalBuffer(_ vector:[MetalComplexNumberType], metalDevice:MTLDevice) -> MTLBuffer {
    var vector = vector
    let byteLength = vector.count*sizeof(MetalComplexNumberType.self) // or size of and actual 1st element object?
    return metalDevice.newBuffer(withBytes: &vector, length: byteLength, options: MTLResourceOptions())
}

func createShaderParametersMetalBuffer(_ shaderParameters:MetalShaderParameters,  metalDevice:MTLDevice) -> MTLBuffer {
    var shaderParameters = shaderParameters
    let byteLength = sizeof(MetalShaderParameters.self)
    return metalDevice.newBuffer(withBytes: &shaderParameters, length: byteLength, options: MTLResourceOptions())
}

func createMatrixShaderParametersMetalBuffer(_ params: MetalMatrixVectorParameters,  metalDevice: MTLDevice) -> MTLBuffer {
    var params = params
    let byteLength = sizeof(MetalMatrixVectorParameters.self)
    return metalDevice.newBuffer(withBytes: &params, length: byteLength, options: MTLResourceOptions())
    
}

func createPoolingParametersMetalBuffer(_ params: MetalPoolingParameters, metalDevice: MTLDevice) -> MTLBuffer {
    var params = params
    let byteLength = sizeof(MetalPoolingParameters.self)
    return metalDevice.newBuffer(withBytes: &params, length: byteLength, options: MTLResourceOptions())
}

func createConvolutionParametersMetalBuffer(_ params: MetalConvolutionParameters, metalDevice: MTLDevice) -> MTLBuffer {
    var params = params
    let byteLength = sizeof(MetalConvolutionParameters.self)
    return metalDevice.newBuffer(withBytes: &params, length: byteLength, options: MTLResourceOptions())
}

func createTensorDimensionsVectorMetalBuffer(_ vector: [MetalTensorDimensions], metalDevice: MTLDevice) -> MTLBuffer {
    var vector = vector
    let byteLength = vector.count * sizeof(MetalTensorDimensions.self)
    return metalDevice.newBuffer(withBytes: &vector, length: byteLength, options: MTLResourceOptions())
}

func setupShaderInMetalPipeline(_ shaderName:String, metalDefaultLibrary:MTLLibrary, metalDevice:MTLDevice) -> (shader:MTLFunction?,
    computePipelineState:MTLComputePipelineState?,
    computePipelineErrors:NSErrorPointer?)  {
        let shader = metalDefaultLibrary.newFunction(withName: shaderName)
        let computePipeLineDescriptor = MTLComputePipelineDescriptor()
        computePipeLineDescriptor.computeFunction = shader
        //        var computePipelineErrors = NSErrorPointer()
        //            let computePipelineState:MTLComputePipelineState = metalDevice.newComputePipelineStateWithFunction(shader!, completionHandler: {(})
        let computePipelineErrors: NSErrorPointer = nil
        var computePipelineState:MTLComputePipelineState? = nil
        do {
            computePipelineState = try metalDevice.newComputePipelineState(with: shader!)
        } catch {
            print("catching..")
        }
        return (shader, computePipelineState, computePipelineErrors)
        
}

func createMetalBuffer(_ vector:[Float], metalDevice:MTLDevice) -> MTLBuffer {
    var vector = vector
    let byteLength = vector.count*sizeof(Float.self)
    return metalDevice.newBuffer(withBytes: &vector, length: byteLength, options: MTLResourceOptions())
}

func preLoadMetalShaders(_ metalDevice: MTLDevice, metalDefaultLibrary: MTLLibrary) {
    let shaders = ["avg_pool", "max_pool", "rectifier_linear", "convolution_layer", "im2col"]
    for shader in shaders {
        setupShaderInMetalPipeline(shader, metalDefaultLibrary: metalDefaultLibrary,metalDevice: metalDevice) // TODO: this returns stuff
    }
}

func createOrReuseFloatMetalBuffer(_ name:String, data: [Float], cache:inout [Dictionary<String,MTLBuffer>], layer_number:Int, metalDevice:MTLDevice) -> MTLBuffer {
    var result:MTLBuffer
    if let tmpval = cache[layer_number][name] {
        print("found key = \(name) in cache")
        result = tmpval
    } else {
        print("didnt find key = \(name) in cache")
        result = createFloatMetalBuffer(data, metalDevice: metalDevice)
        cache[layer_number][name] = result
        // print("DEBUG: cache = \(cache)")
    }
    
    return result
}


func createOrReuseConvolutionParametersMetalBuffer(_ name:String,
    data: MetalConvolutionParameters,
    cache:inout [Dictionary<String,MTLBuffer>], layer_number: Int, metalDevice: MTLDevice) -> MTLBuffer {
        var result:MTLBuffer
        if let tmpval = cache[layer_number][name] {
           print("found key = \(name) in cache")
            result = tmpval
        } else {
            print("didnt find key = \(name) in cache")
            result = createConvolutionParametersMetalBuffer(data, metalDevice: metalDevice)
            cache[layer_number][name] = result
            //print("DEBUG: cache = \(cache)")
        }
        
        return result
}

func createOrReuseTensorDimensionsVectorMetalBuffer(_ name:String,
    data:[MetalTensorDimensions],cache:inout [Dictionary<String,MTLBuffer>], layer_number: Int, metalDevice: MTLDevice) -> MTLBuffer {
        var result:MTLBuffer
        if let tmpval = cache[layer_number][name] {
            print("found key = \(name) in cache")
            result = tmpval
        } else {
            print("didnt find key = \(name) in cache")
            result = createTensorDimensionsVectorMetalBuffer(data, metalDevice: metalDevice)
            cache[layer_number][name] = result
            //print("DEBUG: cache = \(cache)")
        }
        
        return result
}

//
//let sizeParamMetalBuffer = createShaderParametersMetalBuffer(size_params, metalDevice: metalDevice)
//let poolingParamMetalBuffer = createPoolingParametersMetalBuffer(pooling_params, metalDevice: metalDevice)

func createOrReuseShaderParametersMetalBuffer(_ name:String,
    data:MetalShaderParameters,cache:inout [Dictionary<String,MTLBuffer>], layer_number: Int, metalDevice: MTLDevice) -> MTLBuffer {
        var result:MTLBuffer
        if let tmpval = cache[layer_number][name] {
//            print("found key = \(name) in cache")
            result = tmpval
        } else {
//            print("didnt find key = \(name) in cache")
            result = createShaderParametersMetalBuffer(data, metalDevice: metalDevice)
            cache[layer_number][name] = result
            //print("DEBUG: cache = \(cache)")
        }
        
        return result
}

func createOrReusePoolingParametersMetalBuffer(_ name:String,
    data:MetalPoolingParameters,cache:inout [Dictionary<String,MTLBuffer>], layer_number: Int, metalDevice: MTLDevice) -> MTLBuffer {
        var result:MTLBuffer
        if let tmpval = cache[layer_number][name] {
//            print("found key = \(name) in cache")
            result = tmpval
        } else {
//            print("didnt find key = \(name) in cache")
            result = createPoolingParametersMetalBuffer(data, metalDevice: metalDevice)
            cache[layer_number][name] = result
            //print("DEBUG: cache = \(cache)")
        }
        
        return result
}


