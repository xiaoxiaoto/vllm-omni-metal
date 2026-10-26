// SPDX-License-Identifier: Apache-2.0
import Foundation
import CoreML
import Metal

private final class Session {
    let model: MLModel
    let input: MLMultiArray
    let weights: [String:MLMultiArray]
    let output: MLMultiArray
    let provider: MLDictionaryFeatureProvider
    let options: MLPredictionOptions
    init(_ url: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: url, configuration: config)
        guard let ins = model.modelDescription.inputDescriptionsByName["x"]?.multiArrayConstraint,
              let outs = model.modelDescription.outputDescriptionsByName["out"]?.multiArrayConstraint,
              ins.dataType == .float16, outs.dataType == .float16 else {
            throw NSError(domain:"FFN", code:1)
        }
        input = try MLMultiArray(shape: ins.shape, dataType: .float16)
        output = try MLMultiArray(shape: outs.shape, dataType: .float16)
        var arrays: [String:MLMultiArray] = [:]
        var features = ["x":MLFeatureValue(multiArray:input)]
        for (name,desc) in model.modelDescription.inputDescriptionsByName where name != "x" {
            guard ["wg","wu","wd"].contains(name), let constraint=desc.multiArrayConstraint,
                  constraint.dataType == .float16 else { throw NSError(domain:"FFN",code:2) }
            let array=try MLMultiArray(shape:constraint.shape,dataType:.float16)
            arrays[name]=array
            features[name]=MLFeatureValue(multiArray:array)
        }
        weights=arrays
        provider = try MLDictionaryFeatureProvider(dictionary:features)
        options = MLPredictionOptions()
        options.outputBackings = ["out":output]
    }
}
private var sessions: [Int32:Session] = [:]
private var nextHandle: Int32 = 0
private let lock = NSLock()
private let identity = strdup("\(MTLCreateSystemDefaultDevice()?.registryID ?? 0)|\(MTLCreateSystemDefaultDevice()?.name ?? "none")|\(ProcessInfo.processInfo.processorCount)|\(ProcessInfo.processInfo.physicalMemory)")!

@_cdecl("ffn_device")
public func ffnDevice() -> UnsafePointer<CChar> { UnsafePointer(identity) }
private final class PlanResult: @unchecked Sendable { var value: Int32 = -1 }

@_cdecl("ffn_plan")
public func ffnPlan(_ path: UnsafePointer<CChar>) -> Int32 {
    let url = URL(fileURLWithPath:String(cString:path))
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine
    let semaphore = DispatchSemaphore(value:0)
    let result = PlanResult()
    Task.detached {
        defer { semaphore.signal() }
        do {
            let plan = try await MLComputePlan.load(contentsOf:url, configuration:config)
            guard case .program(let program) = plan.modelStructure else { return }
            var total = 0
            var ane = 0
            for function in program.functions.values {
                for op in function.block.operations where op.operatorName.contains("matmul") {
                    total += 1
                    if let usage = plan.deviceUsage(for:op), case .neuralEngine = usage.preferred { ane += 1 }
                }
            }
            result.value = total > 0 && total == ane ? 1 : 0
        } catch { fputs("CoreML plan error: \(error)\n", stderr) }
    }
    guard semaphore.wait(timeout:.now()+60) == .success else { return -2 }
    return result.value
}

@_cdecl("ffn_load")
public func ffnLoad(_ path:UnsafePointer<CChar>, _ rows:Int64, _ hidden:Int64) -> Int32 {
    lock.lock(); defer { lock.unlock() }
    do {
        let session = try Session(URL(fileURLWithPath:String(cString:path)))
        guard rows > 0, hidden > 0, rows <= Int64.max / hidden,
              session.input.count == rows * hidden, session.output.count == rows * hidden,
              nextHandle < Int32.max else { return -2 }
        let index = nextHandle
        nextHandle += 1
        sessions[index] = session
        return index
    } catch { fputs("CoreML load error: \(error)\n", stderr); return -1 }
}

// Separate ABI: the legacy FFN loader retains its square-output contract.
@_cdecl("projection_load")
public func projectionLoad(_ path:UnsafePointer<CChar>, _ rows:Int64, _ inputs:Int64, _ outputs:Int64) -> Int32 {
    lock.lock(); defer { lock.unlock() }
    do {
        let session = try Session(URL(fileURLWithPath:String(cString:path)))
        guard rows > 0, inputs > 0, outputs > 0,
              rows <= Int64.max / inputs, rows <= Int64.max / outputs,
              session.weights.isEmpty,
              session.input.count == rows * inputs, session.output.count == rows * outputs,
              nextHandle < Int32.max else { return -2 }
        let index = nextHandle
        nextHandle += 1
        sessions[index] = session
        return index
    } catch { fputs("CoreML projection load error: \(error)\n", stderr); return -1 }
}

@_cdecl("ffn_unload")
public func ffnUnload(_ index:Int32) {
    lock.lock(); defer { lock.unlock() }
    sessions.removeValue(forKey:index)
}

@_cdecl("ffn_stage")
public func ffnStage(_ index:Int32, _ part:Int32, _ source:UnsafeRawPointer, _ count:Int64) -> Int32 {
    lock.lock(); defer { lock.unlock() }
    guard 0 <= part, part < 3, let session=sessions[index],
          let array=session.weights[["wg","wu","wd"][Int(part)]], Int64(array.count)==count else { return -1 }
    memcpy(array.dataPointer,source,array.count*2)
    return 0
}

@_cdecl("ffn_predict")
public func ffnPredict(_ index:Int32, _ input:UnsafeRawPointer, _ output:UnsafeMutableRawPointer, _ inputCount:Int64, _ outputCount:Int64) -> Int32 {
    lock.lock(); defer { lock.unlock() }
    guard let session = sessions[index], Int64(session.input.count)==inputCount,
          Int64(session.output.count)==outputCount else { return -1 }
    do {
        memcpy(session.input.dataPointer, input, session.input.count*2)
        let result = try session.model.prediction(from:session.provider, options:session.options)
        guard let actual = result.featureValue(for:"out")?.multiArrayValue,
              actual.dataType == .float16, actual.count == session.output.count else { return -2 }
        var stride = 1
        for i in (0..<actual.shape.count).reversed() {
            guard actual.strides[i].intValue == stride else { return -3 }
            stride *= actual.shape[i].intValue
        }
        memcpy(output, actual.dataPointer, actual.count*2)
        return 0
    } catch { fputs("CoreML prediction error: \(error)\n", stderr); return -4 }
}
