package edu.berkeley.cs.amplab.mlmatrix

import scala.io.Source._

import breeze.linalg._
import breeze.numerics._

import edu.berkeley.cs.amplab.mlmatrix.util.QRUtils
import edu.berkeley.cs.amplab.mlmatrix.util.Utils

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._


object Fusion extends Logging with Serializable {


  def solveForX(A: RowPartitionedMatrix, b: RowPartitionedMatrix, solver: String,
      lambda: Double,
      numIterations: Integer,
      stepSize: Double,
      miniBatchFraction: Double) = {
    solver.toLowerCase match {
      case "normal" =>
        new NormalEquations().solveLeastSquaresWithL2(A, b, lambda)
      case "sgd" =>
        new LeastSquaresGradientDescent(numIterations, stepSize, miniBatchFraction).solveLeastSquaresWithL2(A, b, lambda)
      case "tsqr" =>
        new TSQR().solveLeastSquaresWithL2(A, b, lambda)
      case "local" =>
        // Solve regularized least squares problem with local qr factorization
        val (aTilde, bTilde) = QRUtils.qrSolveWithL2(A.collect(), b.collect(), lambda)
        aTilde \ bTilde
      case _ =>
        logError("Invalid Solver " + solver + " should be one of tsqr|normal|sgd")
        logError("Using TSQR")
        new TSQR().solveLeastSquares(A, b)
    }
  }

  def computeResidualNorm(A: RowPartitionedMatrix,
      b: RowPartitionedMatrix,
      xComputed: DenseMatrix[Double]) = {
    val xBroadcast = A.rdd.context.broadcast(xComputed)
    val axComputed = A.mapPartitions { part =>
      part*xBroadcast.value
    }
    val residualNorm = (b - axComputed).normFrobenius()
    residualNorm
  }

  def computeResidualNormWithL2(A: RowPartitionedMatrix,
      b: RowPartitionedMatrix,
      xComputed: DenseMatrix[Double], lambda: Double) = {
    val unregularizedNorm = computeResidualNorm(A,b,xComputed)
    val normX = norm(xComputed.toDenseVector)

    scala.math.sqrt(unregularizedNorm*unregularizedNorm + lambda*normX*normX)
  }


  def loadMatrixFromFile(sc: SparkContext, filename: String):
    RowPartitionedMatrix = {
    RowPartitionedMatrix.fromArray(
      sc.textFile(filename).map(line=>line.split(",")).map(x =>
        x.map(y=> y.toDouble)))
  }

  def getErrPercent(predicted: RDD[Array[Int]], actual: RDD[Array[Int]], numTestImages: Int): Double = {
    // FIXME: Each image only has one actual label, so actual should be an RDD[Int]
    val totalErr = predicted.zip(actual).map({ case (topKLabels, actualLabel) =>
      if (topKLabels.contains(actualLabel(0))) {
        0.0
      } else {
        1.0
      }
    }).reduce(_ + _)

    val errPercent = totalErr / numTestImages * 100.0
    errPercent
  }

  def topKClassifier(k: Int, in: RDD[DenseVector[Double]]) : RDD[Array[Int]] = {
    // Returns top k indices with maximum value
    in.map { ary => argtopk(ary, k).toArray}
  }


  def calcTestErr(daisyTest: RowPartitionedMatrix, lcsTest: RowPartitionedMatrix,
    daisyX: DenseMatrix[Double], lcsX: DenseMatrix[Double],
    actualLabels: RDD[Array[Int]],
    daisyWt: Double, lcsWt: Double): Double = {

    // Compute number of test images
    val numTestImages = daisyTest.numRows().toInt

    // Broadcast x
    val daisyXBroadcast = daisyTest.rdd.context.broadcast(daisyX)
    val lcsXBroadcast = lcsTest.rdd.context.broadcast(lcsX)

    // Calculate predictions
    val daisyPrediction = daisyTest.rdd.map { mat =>
      mat.mat * daisyXBroadcast.value
    }
    val lcsPrediction = lcsTest.rdd.map { mat =>
      mat.mat * lcsXBroadcast.value
    }

    // Fuse b matrices
    val fusedPrediction = daisyPrediction.zip(lcsPrediction).map { p =>
      (p._1*daisyWt + p._2*lcsWt).toDenseVector
    }

    val predictedLabels = topKClassifier(5, fusedPrediction)
    val errPercent = getErrPercent(predictedLabels, actualLabels, numTestImages)
    errPercent
  }

  def main(args: Array[String]) {
    if (args.length < 2) {
      println("Usage: Fusion <master> <solver: tsqr|normal|sgd|local> <lambda> [<stepsize> <numIters> <miniBatchFraction>]")
      System.exit(0)
    }
    val sparkMaster = args(0)
    val solver = args(1)
    // Lambda for regularization
    val lambda = args(2).toDouble

    var stepSize = 0.1
    var numIterations = 10
    var miniBatchFraction = 1.0
    if (solver == "sgd") {
      if (args.length < 6) {
        println("Usage: StabilityChecker <master> <sparkHome> " +
          "<solver: tsqr|normal|sgd|local> [<stepsize> <numIters> <miniBatchFraction>]")
        System.exit(0)
      } else {
        stepSize = args(3).toDouble
        numIterations = args(4).toInt
        miniBatchFraction = args(5).toDouble
      }
    }

    println("Testing the fusion class")

    val conf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName("Fusion")
      .setJars(SparkContext.jarOfClass(this.getClass).toSeq)
    val sc = new SparkContext(conf)

    //Directory that holds the data
    val directory = "imagenet-linear-solver-data/"

    // Daisy filenames
    val daisyTrainFilename = directory + "daisy-aPart1-1/"
    val daisyTestFilename = directory + "daisy-testFeatures-test-1/"
    val daisyBFilename = directory + "daisy-null-labels/"

    // LCS filenames
    val lcsTrainFilename = directory + "lcs-aPart1-1/"
    val lcsTestFilename = directory + "lcs-testFeatures-test-1/"
    val lcsBFilename = directory + "lcs-null-labels/"

    // Actual labels from imagenet
    val imagenetTestLabelsFilename = directory + "imagenet-test-actual/"

    // Load data as RowPartitionedMatrices
    var daisyTrain = loadMatrixFromFile(sc, daisyTrainFilename)
    var daisyTest = loadMatrixFromFile(sc, daisyTestFilename)
    var daisyB = loadMatrixFromFile(sc, daisyBFilename)


    var lcsTrain = loadMatrixFromFile(sc, lcsTrainFilename)
    var lcsTest = loadMatrixFromFile(sc, lcsTestFilename)
    var lcsB = loadMatrixFromFile(sc, lcsBFilename)

    // FIXME: Should Repartition data and labels together to 16 partitions
    val data: RDD[(((((RowPartition, RowPartition), RowPartition), RowPartition), RowPartition), RowPartition)] =
      daisyTrain.rdd.zip(daisyTest.rdd).zip(daisyB.rdd).zip(lcsTrain.rdd).zip(lcsTest.rdd).zip(lcsB.rdd).repartition(16)
    val first: RDD[RowPartition] = data.map(parts => parts._1._1._1._1._1)
    val second: RDD[RowPartition] = data.map(parts => parts._1._1._1._1._2)
    val third: RDD[RowPartition] = data.map(parts => parts._1._1._1._2)
    val fourth: RDD[RowPartition] = data.map(parts => parts._1._1._2)
    val fifth: RDD[RowPartition] = data.map(parts => parts._1._2)
    val sixth: RDD[RowPartition] = data.map(parts => parts._2)
    daisyTrain = new RowPartitionedMatrix(first)
    daisyTest = new RowPartitionedMatrix(second)
    daisyB = new RowPartitionedMatrix(third)
    lcsTrain = new RowPartitionedMatrix(fourth)
    lcsTest = new RowPartitionedMatrix(fifth)
    lcsB = new RowPartitionedMatrix(sixth)


    // Load text file as array of ints
    val imagenetTestLabels = sc.textFile(imagenetTestLabelsFilename).map(line=>line.split(",")).map(x =>
      x.map(y=> y.toInt))

    // Solve for daisy x
    var begin = System.nanoTime()
    val daisyX = solveForX(daisyTrain, daisyB, solver, lambda, numIterations, stepSize, miniBatchFraction)
    var end = System.nanoTime()
    // Timing numbers are in ms
    val daisyTime = (end - begin) / 1e6

    println("Finished solving for daisy X ")

    // Solve for lcs x
    var begin2 = System.nanoTime()
    val lcsX = solveForX(lcsTrain, lcsB, solver, lambda, numIterations, stepSize, miniBatchFraction)
    var end2 = System.nanoTime()
    val lcsTime = (end2 -begin2) /1e6

    // FIXME: Residual norm needs to be calculated for regularized problem

    // Information about the spectrum of the matrices
    // println("Condition number of daisyTrain " + daisyTrain.condEst())
    // println("Condition number of daisyTest " + daisyTest.condEst())
    // println("SVDs of daisyTrain " + daisyTrain.svds().toArray.mkString(" "))
    // println("Condition number of lcsTrain " + lcsTrain.condEst())
    // println("Condition number of lcsTest " + lcsTest.condEst())
    // println("SVDs of lcsTrain " + lcsTrain.svds().toArray.mkString(" "))

    val daisyResidual= computeResidualNormWithL2(daisyTrain, daisyB, daisyX, lambda)
    val lcsResidual = computeResidualNormWithL2(lcsTrain, lcsB, lcsX, lambda)
    val testError = calcTestErr(daisyTest, lcsTest, daisyX, lcsX, imagenetTestLabels, 0.5, 0.5)
    println("Condition number, residual norm, time")
    println("Daisy: ")
    println(daisyTrain.condEst(), daisyResidual, daisyTime)
    println("LCS: ")
    println(lcsTrain.condEst(), lcsResidual, lcsTime)
    println(testError)
  }
}
