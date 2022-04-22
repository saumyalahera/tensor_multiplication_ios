//
//  SLTensor.swift
//  tensor_multiplication_ios
//
//  Created by Saumya Lahera on 4/21/22.
//

import UIKit
import MLCompute

/**Idea is to conver to the MLTensor that will be used for transposition and other computations*/
public class SLTensor: NSObject {
    
    var values:[Float]!
    
    //This will change if you can change the tensor rank
    let rank = 2
    
    //Not needed, it is used to mimic Tenor Flow librray
    var name:String!
    
    //This is used for tensor contraction and reshaping the tensor
    var shape:[Int]!
    
    
//MARK: - Init
    init(values: [Float], shape:[Int], name:String) {
        super.init()
        
        self.values = values
        self.name = name
        self.shape = shape
    }
}
