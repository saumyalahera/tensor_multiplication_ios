//
//  ViewController.swift
//  tensor_multiplication_ios
//
//  Created by Saumya Lahera on 4/21/22.
//

import UIKit

class ViewController: UIViewController {

    typealias tf = SLTensorHelper
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let tensor1 = SLTensor(values: [11,2,3,4], shape: [2,2], name: "t1")
        let tensor2 = SLTensor(values: [1,2,3,4], shape: [2,2], name: "t2")
        
        //Multiplication
        let tensor3 = tf.matmul(tensorA: tensor1, tensorB: tensor2, name: "t3")
        //Assuming there will always be not nil values
        print(tensor3.values!)
    }

}

