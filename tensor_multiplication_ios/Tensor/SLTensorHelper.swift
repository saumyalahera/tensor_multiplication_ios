//
//  SLTensorHelper.swift
//  tensor_multiplication_ios
//
//  Created by Saumya Lahera on 4/21/22.
//

import UIKit
import MLCompute

//MARK: - Operations
class SLTensorHelper: NSObject {

    static func tensorOperation(tensors:[SLTensor], layer:MLCLayer, outputShape: [Int]) -> [Float] {
        let outputCount = outputShape.reduce(1, {$0*$1})
        return tensorsOperation(tensors: tensors, layer: layer, outputCount: outputCount)
    }
    
    //It is the main function for all the math operation because it works with layers for GPU or CPU because it uses Metal Performance Shaders
    /**This is the main method that takes care of all the operations. It needs a MLCompute layer which can be of different kind */
    static func tensorsOperation(tensors:[SLTensor], layer:MLCLayer, outputCount: Int) -> [Float]{
        
        let tensorDataValuesCount = outputCount
        
        var inputs = [String:MLCTensor]()
        var sources = [MLCTensor]()
        var inputsData = [String:MLCTensorData]()
        
        for i in 0..<tensors.count {
            let tensor = MLCTensor(shape: tensors[i].shape, dataType: .float32)
            sources.append(tensor)
            inputs["\(i)"] = tensor
            
            let tensorDataValues = tensors[i].values
            
            //Assuming that there will always be not nil values
            let tensorData = MLCTensorData(immutableBytesNoCopy: UnsafeRawPointer(tensorDataValues!),length: tensorDataValues!.count * MemoryLayout<Float>.size)
            inputsData["\(i)"]=tensorData
        }
       
        
        let tensorsGraph = MLCGraph()
        tensorsGraph.node(with: layer,sources: sources)
       
        let tensorsPlan = MLCInferenceGraph(graphObjects: [tensorsGraph])
        tensorsPlan.addInputs(inputs)
        tensorsPlan.compile(options: .debugLayers, device: MLCDevice())

        var results:[Float] = [0.0]
        
        tensorsPlan.execute(inputsData: inputsData, batchSize: 0, options: []) { (r, e, time) in
            print("Error: \(String(describing: e))")
            print("Result: \(String(describing: r))")
            
            let bufO = UnsafeMutableRawPointer.allocate(byteCount: tensorDataValuesCount * MemoryLayout<Float>.size, alignment: MemoryLayout<Float>.alignment)
            
            r!.copyDataFromDeviceMemory(toBytes: bufO, length: tensorDataValuesCount * MemoryLayout<Float>.size, synchronizeWithDevice: false)
            
            let outArray = bufO.bindMemory(to: Float.self, capacity: tensorDataValuesCount)
            let outArrayDat = UnsafeBufferPointer(start: outArray, count: tensorDataValuesCount)
            //print(Array(outArrayDat))
            results = Array(outArrayDat)
        }
        return results
    }
}

//MARK: - Multiplication
extension SLTensorHelper {
    
    /*This is needed to check the rank for multiplication*/
    static func getMatmulOutputShape(tensorAShape: [Int], tensorBShape: [Int]) -> [Int]{
        var outputShape = tensorAShape //tensorA.shape.reduce(1, {$0*$1}) //This can also work for 2 same shape size values. It is not recommended
        if(outputShape.count == 2) {
            outputShape = [tensorAShape[0],tensorBShape[1]]
        }else if(outputShape.count == 3) {
            outputShape = [tensorAShape[0],tensorAShape[1],tensorBShape[2]]
        }
        return outputShape
    }
    
    /**This will multiply two tensors and return a new tensor**/
    static func matmul(tensorA:SLTensor, tensorB:SLTensor, name:String) -> SLTensor {
        let outputShape = getMatmulOutputShape(tensorAShape: tensorA.shape, tensorBShape: tensorB.shape)
        let outputValues = self.tensorOperation(tensors: [tensorA,tensorB], layer: MLCMatMulLayer(descriptor: MLCMatMulDescriptor())!, outputShape: outputShape)
        return SLTensor(values: outputValues, shape: outputShape, name: name)
    }
    
}


