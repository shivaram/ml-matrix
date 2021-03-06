package edu.berkeley.cs.amplab.mlmatrix.util

import scala.reflect.ClassTag

import org.apache.spark.SparkContext._
import org.apache.spark.HashPartitioner
import org.apache.spark.rdd.RDD

import breeze.linalg._

object Utils {

  /**
   * Deep copy a Breeze matrix
   */
  def cloneMatrix(in: DenseMatrix[Double]) = {
    in.copy
  }

  def decomposeLowerUpper(A: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val L = new DenseMatrix[Double](A.rows, min(A.rows, A.cols))
    val U = new DenseMatrix[Double](min(A.cols, A.rows), A.cols)

    var i = 0
    while(i < A.rows) {
      var j = 0
      while(j < A.cols) {
        if (i < j) {
          U(i, j) =  A(i, j)
        } else if (i == j) {
          U(i, i) = A(i, i)
          L(i, i) = 1.0
        } else {
          L(i, j) = A(i, j)
        }
        j = j + 1
      }
      i = i + 1
    }
    (L, U)
  }

  /**
   * Reduces the elements of this RDD in a multi-level tree pattern.
   *
   * @param depth suggested depth of the tree (default: 2)
   * @see [[org.apache.spark.rdd.RDD#reduce]]
   */
  def treeReduce[T: ClassTag](rdd: RDD[T], f: (T, T) => T, depth: Int = 2): T = {
    require(depth >= 1, s"Depth must be greater than or equal to 1 but got $depth.")
    val reducePartition: Iterator[T] => Option[T] = iter => {
      if (iter.hasNext) {
        Some(iter.reduceLeft(f))
      } else {
        None
      }
    }
    val partiallyReduced = rdd.mapPartitions(it => Iterator(reducePartition(it)))
    val op: (Option[T], Option[T]) => Option[T] = (c, x) => {
      if (c.isDefined && x.isDefined) {
        Some(f(c.get, x.get))
      } else if (c.isDefined) {
        c
      } else if (x.isDefined) {
        x
      } else {
        None
      }
    }
    treeAggregate(Option.empty[T])(partiallyReduced, op, op, depth)
      .getOrElse(throw new UnsupportedOperationException("empty collection"))
  }

  /**
   * Aggregates the elements of this RDD in a multi-level tree pattern.
   *
   * @param depth suggested depth of the tree (default: 2)
   * @see [[org.apache.spark.rdd.RDD#aggregate]]
   */
  def treeAggregate[T: ClassTag, U: ClassTag](zeroValue: U)(
      rdd: RDD[T],
      seqOp: (U, T) => U,
      combOp: (U, U) => U,
      depth: Int = 2): U = {
    require(depth >= 1, s"Depth must be greater than or equal to 1 but got $depth.")
    if (rdd.partitions.size == 0) {
      return zeroValue
    }
    val aggregatePartition = (it: Iterator[T]) => it.aggregate(zeroValue)(seqOp, combOp)
    var partiallyAggregated = rdd.mapPartitions(it => Iterator(aggregatePartition(it)))
    var numPartitions = partiallyAggregated.partitions.size
    val scale = math.max(math.ceil(math.pow(numPartitions, 1.0 / depth)).toInt, 2)
    // If creating an extra level doesn't help reduce the wall-clock time, we stop tree aggregation.
    while (numPartitions > 1) { // while (numPartitions > scale + numPartitions / scale) {
      numPartitions /= scale
      val curNumPartitions = numPartitions
      partiallyAggregated = partiallyAggregated.mapPartitionsWithIndex { (i, iter) =>
        iter.map((i % curNumPartitions, _))
      }.reduceByKey(new HashPartitioner(curNumPartitions), combOp).values
    }
    partiallyAggregated.reduce(combOp)
  }

  def aboutEq(a: DenseMatrix[Double], b: DenseMatrix[Double], thresh: Double = 1e-8) = {
    math.abs(max(a-b)) < thresh
  }
}
