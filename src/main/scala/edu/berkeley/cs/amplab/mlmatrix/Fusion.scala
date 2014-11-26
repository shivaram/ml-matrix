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

  def loadMatrixFromFile(sc: SparkContext, filename: String):
    RowPartitionedMatrix = {
    RowPartitionedMatrix.fromArray(
      sc.textFile(filename).map(line=>line.split(",")).map(x =>
        x.map(y=> y.toDouble)))
  }

  def getErrPercent(predicted: RDD[Array[Int]], actual: RDD[Array[Int]], numTestImages: Int): Double = {
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
    // ary.toSeq.zipWithIndex.sortBy(_._1).takeRight(k).map(_._2).toArray
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
      println("Usage: Fusion <master> <solver: tsqr|normal|sgd|local> [<stepsize> <numIters> <miniBatchFraction>]")
      System.exit(0)
    }
    val sparkMaster = args(0)
    val solver = args(1)

    var stepSize = 0.1
    var numIterations = 10
    var miniBatchFraction = 1.0
    if (solver == "sgd") {
      if (args.length < 5) {
        println("Usage: StabilityChecker <master> <sparkHome> " +
          "<solver: tsqr|normal|sgd|local> [<stepsize> <numIters> <miniBatchFraction>]")
        System.exit(0)
      } else {
        stepSize = args(6).toDouble
        numIterations = args(7).toInt
        miniBatchFraction = args(8).toDouble
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
    val daisyTrain = loadMatrixFromFile(sc, daisyTrainFilename)
    val daisyTest = loadMatrixFromFile(sc, daisyTestFilename)
    val daisyB = loadMatrixFromFile(sc, daisyBFilename)

    val lcsTrain = loadMatrixFromFile(sc, lcsTrainFilename)
    val lcsTest = loadMatrixFromFile(sc, lcsTestFilename)
    val lcsB = loadMatrixFromFile(sc, lcsBFilename)

    //Load text file as array of ints
    val imagenetTestLabels = sc.textFile(imagenetTestLabelsFilename).map(line=>line.split(",")).map(x =>
      x.map(y=> y.toInt))

    //Solve for daisy x
    var begin = System.nanoTime()
    val daisyX = solver.toLowerCase match {
      case "normal" =>
        new NormalEquations().solveLeastSquares(daisyTrain, daisyB)
      case "sgd" =>
        new LeastSquaresGradientDescent(numIterations, stepSize, miniBatchFraction).solveLeastSquares(daisyTrain, daisyB)
      case "tsqr" =>
        println("Size of daisyTrain is " + daisyTrain.getDim())
        println("Size of daisyB is " + daisyB.getDim())
        new TSQR().solveLeastSquares(daisyTrain, daisyB)
      case "local" =>
        val (r, qtb) = QRUtils.qrSolve(daisyTrain.collect(), daisyB.collect())
        r \ qtb
      case _ =>
        logError("Invalid Solver " + solver + " should be one of tsqr|normal|sgd")
        logError("Using TSQR")
        new TSQR().solveLeastSquares(daisyTrain, daisyB)
    }
    var end = System.nanoTime()
    // Timing numbers are in ms
    val daisyTime = (end - begin) / 1e6

    //Solve for lcs x
    var begin2 = System.nanoTime()
    val lcsX = solver.toLowerCase match {
      case "normal" =>
        new NormalEquations().solveLeastSquares(lcsTrain, lcsB)
      case "sgd" =>
        new LeastSquaresGradientDescent(numIterations, stepSize, miniBatchFraction).solveLeastSquares(lcsTrain, lcsB)
      case "tsqr" =>
        new TSQR().solveLeastSquares(lcsTrain, lcsB)
      case "local" =>
        val (r, qtb) = QRUtils.qrSolve(lcsTrain.collect(), lcsB.collect())
        r \ qtb
      case _ =>
        logError("Invalid Solver " + solver + " should be one of tsqr|normal|sgd")
        logError("Using TSQR")
        new TSQR().solveLeastSquares(lcsTrain, lcsB)
    }
    var end2 = System.nanoTime()
    val lcsTime = (end2 -begin2) /1e6


    // Information about the spectrum of the matrices
    // println("Condition number of daisyTrain " + daisyTrain.condEst())
    // println("Condition number of daisyTest " + daisyTest.condEst())
    // println("SVDs of daisyTrain " + daisyTrain.svds().toArray.mkString(" "))
    // println("Condition number of lcsTrain " + lcsTrain.condEst())
    // println("Condition number of lcsTest " + lcsTest.condEst())
    // println("SVDs of lcsTrain " + lcsTrain.svds().toArray.mkString(" "))

    val daisyResidualNorm = computeResidualNorm(daisyTrain, daisyB, daisyX)
    val lcsResidualNorm = computeResidualNorm(lcsTrain, lcsB, lcsX)
    val testError = calcTestErr(daisyTest, lcsTest, daisyX, lcsX, imagenetTestLabels, 0.5, 0.5)
    // println("Condition number, residual norm, time")
    // println("Daisy: ")
    // println(daisyTrain.condEst(), daisyResidualNorm, daisyTime)
    // println("LCS: ")
    // println(lcsTrain.condEst(), lcsResidualNorm, lcsTime)



    println(testError)
  }
}
