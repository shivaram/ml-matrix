package edu.berkeley.cs.amplab.mlmatrix

import java.io.ObjectOutputStream
import java.io.IOException
import java.util.concurrent.ThreadLocalRandom
import scala.collection.mutable.ArrayBuffer

import breeze.linalg._
import breeze.numerics._

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.Partition
import org.apache.spark.NarrowDependency
import org.apache.spark.Dependency
import org.apache.spark.TaskContext

import edu.berkeley.cs.amplab.mlmatrix.util.Utils

class ColumnBlockMultiplyPartition(
    idx: Int,
    colIdx: Int,
    colRDD: RDD[_]) extends Partition {
  var colPart = colRDD.partitions(colIdx)
  override val index: Int = idx

  @throws(classOf[IOException])
  private def writeObject(oos: ObjectOutputStream): Unit =  {
    // Update the reference to parent split at the time of task serialization
    colPart = colRDD.partitions(colIdx)
    oos.defaultWriteObject()
  }
}

// Multiplies RDD[RowPartition].t * blockRDD
// Expects that firstColBlock and otherColsBlocked have same number of row partitions
// and same number of rows per partition.
class ColumnBlockMultiplyRDD(
    sc: SparkContext,
    var columnRDD: RDD[RowPartition],
    var blockRDD: RDD[BlockPartition],
    @transient var partitionIdMap: Map[Int, Int])
  extends RDD[BlockPartition](sc, Nil) with Serializable {

  override def getPartitions: Array[Partition] = {
    blockRDD.partitions.map { x =>
      new ColumnBlockMultiplyPartition(x.index, partitionIdMap(x.index), columnRDD)
    }.toArray
  }

  override def getPreferredLocations(split: Partition): Seq[String] = {
    blockRDD.preferredLocations(split)
  }

  override def compute(split: Partition, context: TaskContext) = {
    val currSplit = split.asInstanceOf[ColumnBlockMultiplyPartition]

    // What we need here is to get 1 partition of block and 1 partition
    // of column and multiply then
    val blockPartIter = blockRDD.iterator(currSplit, context)
    val colPartIter = columnRDD.iterator(currSplit.colPart, context)

    blockPartIter.zip(colPartIter).map { case (b, c) =>
      var begin = System.nanoTime()
      val res = c.mat.t * b.mat
      var end = System.nanoTime()
      println("BLAS took " + (end - begin)/1e6 + " ms")
      new BlockPartition(b.blockIdRow, b.blockIdCol, res)
    }
  }

  override def getDependencies: Seq[Dependency[_]] = List(
    new NarrowDependency(blockRDD) {
      def getParents(id: Int): Seq[Int] = List(id)
    },
    new NarrowDependency(columnRDD) {
      def getParents(id: Int): Seq[Int] = List(partitionIdMap(id))
    }
  )

  override def clearDependencies() {
    super.clearDependencies()
    blockRDD = null
    columnRDD = null
  }
}

object BlockMultiplyUsingRDD extends Logging {

  // Multiplies firstColBlock.t * otherColsBlocked
  def doMultiply(
      firstColBlock: RowPartitionedMatrix,
      otherColsBlocked: BlockPartitionedMatrix) = {

    var begin = System.nanoTime()

    val colPartitions = firstColBlock.getPartitionInfo
    val blockPartitions = otherColsBlocked.getBlockInfo

    // Build a map from block partition id to firstColBlockPartitionId
    val colBlocksToPartitions = colPartitions.map { x =>
      // Only works if one block per partition
      assert(x._2.length == 1)
      (x._2(0).blockId, x._2(0).partitionId)
    }.toMap

    val partitionIdMap = blockPartitions.map { x =>
      (x._2.partitionId, colBlocksToPartitions(x._2.blockIdRow))
    }.toMap

    val res = new ColumnBlockMultiplyRDD(firstColBlock.rdd.context,
      firstColBlock.rdd, otherColsBlocked.rdd, partitionIdMap)

    val colSums = res.map(x => (x.blockIdCol, x.mat)).reduceByKey { case (x, y) =>
      x + y
    }

    colSums.cache().count
    var end = System.nanoTime()

    println("BlockQRMultiply took " + (end - begin)/1e6 + " ms")
    colSums
  }

  // Example invocation 
  // ./sbt/sbt assembly
  // ./run-main.sh edu.berkeley.cs.amplab.mlmatrix.BlockMultiplyUsingRDD "local[4]" 1000 200 250 100 true
  def main(args: Array[String]) {
    if (args.length < 4) {
      println("Usage: BlockMultiplyUsingRDD <master> <numRows> <numCols> <rowsPerPart> <colsPerPart> [verify=false]")
      System.exit(0)
    }

    val sparkMaster = args(0)
    val numRows = args(1).toInt
    val numCols = args(2).toInt
    val rowsPerPart = args(3).toInt
    val colsPerPart = args(4).toInt
    val verify = if (args.length > 5) args(5).toBoolean else false

    val conf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName("BlockMultiplyUsingRDD")
      .setJars(SparkContext.jarOfClass(this.getClass).toSeq)
    val sc = new SparkContext(conf)

    Thread.sleep(5000)

    val matrixRDD = sc.parallelize(1 to numRows, numRows / rowsPerPart).map { row =>
      Array.fill(numCols)(ThreadLocalRandom.current().nextGaussian())
    }

    matrixRDD.cache().count()

    val A = BlockPartitionedMatrix.fromArray(matrixRDD, rowsPerPart, colsPerPart)
    A.cache()
    val dim = A.getDim

    val firstColBlock = A.getColBlock(0).cache()
    val otherColsBlocked = A.getBlockRange(0, A.numRowBlocks,
      1, A.numColBlocks).cache()

    firstColBlock.rdd.count()
    otherColsBlocked.rdd.count()

    val computed = doMultiply(firstColBlock, otherColsBlocked)

    // Code to check results !
    if (verify) {
      val actual = firstColBlock.collect().t * otherColsBlocked.collect()
      val computedMat = computed.map(x => x._2).collect().reduceLeftOption((a, b) => DenseMatrix.horzcat(a,
        b)).getOrElse(new DenseMatrix[Double](0, 0))

      // Check dims first
      println("DIMS computed " + computedMat.rows + "x" + computedMat.cols +
        " actual " + actual.rows + "x" + actual.cols)
      assert(computedMat.rows == actual.rows)
      assert(computedMat.cols == actual.cols)
      // Now the matrix
      println("NORM2 computed " + norm(computedMat.toDenseVector) + " " +
        norm(actual.toDenseVector))
      assert(max(abs(computedMat - actual)) < 1e-5)
    }

    sc.stop()
    System.exit(0)
  }
}
