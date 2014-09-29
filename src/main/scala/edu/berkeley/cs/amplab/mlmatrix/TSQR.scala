package edu.berkeley.cs.amplab.mlmatrix

import java.util.concurrent.ThreadLocalRandom
import scala.collection.mutable.ArrayBuffer

import breeze.linalg._

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import edu.berkeley.cs.amplab.mlmatrix.util.QRUtils

class TSQR extends Logging with Serializable {

  def qrR(mat: RowPartitionedMatrix): DenseMatrix[Double] = {
    var qrTree = mat.rdd.mapPartitionsWithIndex { case (idx, part) =>
      val qrResult = QRUtils.qrR(part.next.mat)
      Iterator((idx, qrResult))
    }
    var numParts = qrTree.partitions.length.toDouble
    // TODO: Use aggregateTree from RDD API
    while (numParts > 1) {
      qrTree = qrTree.map(x => ((x._1/2.0).toInt, x._2)).reduceByKey(
        numPartitions=(numParts/2.0).toInt,
        func=reduceQR(_, _))
      numParts = math.ceil(numParts/2.0).toInt
    }
    val result = qrTree.map(x => x._2)
    result.collect()(0)
  }

  private def reduceQR(a: DenseMatrix[Double], b: DenseMatrix[Double]): DenseMatrix[Double] = {
    QRUtils.qrR(DenseMatrix.vertcat(a, b))
  }

  def qrQR(mat: RowPartitionedMatrix): (RowPartitionedMatrix, DenseMatrix[Double]) = {
    // First step run TSQR, get YTR tree
    val (qrTree, r) = qrYTR(mat)

    var curTreeIdx = qrTree.size - 1

    // Now construct Q by going up the tree
    var qrRevTree = qrTree(curTreeIdx).map { part =>
      val yPart = part._2._1
      val tPart = part._2._2
      val qIn = new DenseMatrix[Double](yPart.rows, yPart.cols)
      for (i <- 0 until yPart.cols) {
        qIn(i, i) =  1.0
      }
      (part._1, QRUtils.applyQ(yPart, tPart, qIn, transpose=false))
    }.flatMap { x =>
      val nrows = x._2.rows
      Iterator((x._1 * 2, x._2(0 until nrows/2, ::)),
               (x._1 * 2 + 1, x._2(nrows/2 until nrows, ::)))
    }

    var prevTree = qrRevTree

    while (curTreeIdx > 0) {
      curTreeIdx = curTreeIdx - 1
      prevTree = qrRevTree
      if (curTreeIdx > 0) {
        val nextNumParts = qrTree(curTreeIdx - 1).partitions.size
        qrRevTree = qrTree(curTreeIdx).join(prevTree).flatMap { part =>
          val yPart = part._2._1._1
          val tPart = part._2._1._2
          val qPart = part._2._2
          if (part._1 * 2 + 1 < nextNumParts) {
            val qOut = QRUtils.applyQ(yPart, tPart, qPart, transpose=false)
            val nrows = qOut.rows
            Iterator((part._1 * 2, qOut(0 until nrows/2, ::)),
                     (part._1 * 2 + 1, qOut(nrows/2 until nrows, ::)))
          } else {
            Iterator((part._1 * 2, qPart))
          }
        }
      } else {
        qrRevTree = qrTree(curTreeIdx).join(prevTree).map { part =>
          val yPart = part._2._1._1
          val tPart = part._2._1._2
          val qPart = part._2._2
          (part._1, QRUtils.applyQ(yPart, tPart, qPart, transpose=false))
        }
      }
    }
    (RowPartitionedMatrix.fromMatrix(qrRevTree.map(x => x._2)), r)
  }

  private def qrYTR(mat: RowPartitionedMatrix):
      (Seq[RDD[(Int, (DenseMatrix[Double], Array[Double], DenseMatrix[Double]))]],
        DenseMatrix[Double]) = {
    val qrTreeSeq = new ArrayBuffer[RDD[(Int, (DenseMatrix[Double], Array[Double], DenseMatrix[Double]))]]
    var qrTree = mat.rdd.mapPartitionsWithIndex { case (idx, part) =>
      if (part.hasNext) {
        val qrResult = QRUtils.qrYTR(part.next.mat)
        Iterator((idx, qrResult))
      } else {
        Iterator()
      }
    }
    qrTreeSeq.append(qrTree)

    var numParts = qrTree.partitions.length.toDouble
    while (numParts > 1) {
      qrTree = qrTree.map(x => ((x._1/2.0).toInt, x._2)).reduceByKey(
        numPartitions=math.ceil(numParts/2.0).toInt,
        func=reduceYTR(_, _))
      numParts = math.ceil(numParts/2.0).toInt
      qrTreeSeq.append(qrTree)
    }
    val r = qrTree.map(x => x._2._3).collect()(0)
    (qrTreeSeq, r)
  }

  private def reduceYTR(
      a: (DenseMatrix[Double], Array[Double], DenseMatrix[Double]),
      b: (DenseMatrix[Double], Array[Double], DenseMatrix[Double])) 
    : (DenseMatrix[Double], Array[Double], DenseMatrix[Double]) = {
    QRUtils.qrYTR(DenseMatrix.vertcat(a._3, b._3))
  }

  def solveLeastSquares(
      A: RowPartitionedMatrix,
      b: RowPartitionedMatrix): DenseMatrix[Double] = {
    solveLeastSquaresWithL2(A, b, 0.0)
  }

  def solveLeastSquaresWithL2(
      A: RowPartitionedMatrix,
      b: RowPartitionedMatrix,
      lambda: Double): DenseMatrix[Double] = {
    solveLeastSquaresWithManyL2(A, b, Array(lambda)).head
  }

  // From http://math.stackexchange.com/questions/299481/qr-factorization-for-ridge-regression
  // To solve QR with L2, we need to factorize \pmatrix{ A \\ \Gamma}
  // i.e. A and \Gamma stacked vertically, where \Gamma is a nxn Matrix.
  // To do this we first use TSQR on A and then locally stack \Gamma below and recompute QR.
  def solveLeastSquaresWithManyL2(
      A: RowPartitionedMatrix,
      b: RowPartitionedMatrix,
      lambdas: Array[Double]): Seq[DenseMatrix[Double]] = {
    val matrixParts = A.rdd.zip(b.rdd).map(x => (x._1.mat, x._2.mat))
    var qrTree = matrixParts.mapPartitionsWithIndex { case (idx, part) =>
      val (aPart, bPart) = part.next
      if (aPart.rows < aPart.cols) {
        Iterator((idx, (aPart, bPart)))
      } else {
        val (rPart, bUpdated) = QRUtils.qrSolve(aPart, bPart)
        Iterator((idx, (rPart, bUpdated)))
      }
    }
    var numParts = qrTree.partitions.length.toDouble
    while (numParts > 1) {
      qrTree = qrTree.map(x => ((x._1/2.0).toInt, x._2)).reduceByKey(
        numPartitions=(numParts/2.0).toInt,
        func=reduceQRSolve(_, _))
      numParts = math.ceil(numParts/2.0).toInt
    }

    val results = lambdas.map { lambda =>
      // We only have one partition right now
      val result = qrTree.map { x =>
        val (rFinal, bFinal) = x._2
        val out = if (lambda == 0.0) {
          rFinal \ bFinal
        } else {
          val lambdaRB = (DenseMatrix.eye[Double](rFinal.cols) :* lambda,
            new DenseMatrix[Double](rFinal.cols, bFinal.cols))
          val reduced = reduceQRSolve((rFinal, bFinal), lambdaRB)
          reduced._1 \ reduced._2
        }
        out
      }
      result.collect()(0)
    }
    results
  }

  private def reduceQRSolve(
      a: (DenseMatrix[Double], DenseMatrix[Double]),
      b: (DenseMatrix[Double], DenseMatrix[Double])): (DenseMatrix[Double], DenseMatrix[Double]) = {
    QRUtils.qrSolve(DenseMatrix.vertcat(a._1, b._1),
      DenseMatrix.vertcat(a._2, b._2))
  }

}

object TSQR extends Logging {

  def main(args: Array[String]) {
    if (args.length < 6) {
      println("Usage: TSQR <master> <numRows> <numCols> <numParts> <numClasses>")
      System.exit(0)
    }

    val sparkMaster = args(0)
    val numRows = args(1).toInt
    val numCols = args(2).toInt
    val numParts = args(3).toInt
    val numClasses = args(4).toInt

    val conf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName("TSQR")
      .setJars(SparkContext.jarOfClass(this.getClass).toSeq)
    val sc = new SparkContext(conf)

    val rowsPerPart = numRows / numParts
    val matrixParts = sc.parallelize(1 to numParts, numParts).mapPartitions { part =>
      val data = new Array[Double](rowsPerPart * numCols)
      var i = 0
      while (i < rowsPerPart * numCols) {
        data(i) = ThreadLocalRandom.current().nextGaussian()
        i = i + 1
      }
      val mat = new DenseMatrix[Double](rowsPerPart, numCols, data)
      Iterator(mat)
    }
    matrixParts.cache().count()

    var begin = System.nanoTime()
    val A = RowPartitionedMatrix.fromMatrix(matrixParts)
    val R = new TSQR().qrR(A)
    var end = System.nanoTime()
    logInfo("Random TSQR of " + numRows + "x" + numCols + " took " + (end - begin)/1e6 + "ms")

    // Use the linear solver
    begin = System.nanoTime()
    val b =  A.mapPartitions(
      part => DenseMatrix.rand(part.rows, numClasses)).cache()

    val x = new TSQR().solveLeastSquares(A, b)
    end = System.nanoTime()
    logInfo("Linear solver of " + numRows + "x" + numCols + " took " + (end - begin)/1e6 + "ms")
  }
  
}
